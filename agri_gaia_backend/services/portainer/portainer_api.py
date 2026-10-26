#!/usr/bin/env python

# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
# SPDX-FileContributor: Henri Graf
# SPDX-FileContributor: Jonas Tüpker
# SPDX-FileContributor: Lukas Hesse
# SPDX-FileContributor: Maik Fruhner
# SPDX-FileContributor: Prof. Dr.-Ing. Heiko Tapken
# SPDX-FileContributor: Tobias Wamhof
# SPDX-FileContributor: Philipp Schröer
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

import os
import requests
from requests.exceptions import HTTPError
import datetime
import base64
import json
from typing import Dict, List
import logging

from fastapi.exceptions import HTTPException

from agri_gaia_backend.schemas.container_deployment import ContainerDeployment
from agri_gaia_backend.schemas.container_image import ContainerImage
from agri_gaia_backend.schemas.edge_device import EdgeDevice
from agri_gaia_backend.util.env import bool_from_env

from agri_gaia_backend.util.auth.service_account import (
    REALM_SERVICE_ACCOUNT_USERNAME,
    REALM_SERVICE_ACCOUNT_PASSWORD,
)

logger = logging.getLogger("api-logger")

PROJECT_BASE_URL = os.environ.get("PROJECT_BASE_URL")

PORTAINER_EDGE_PORT = os.environ.get("PORTAINER_EDGE_PORT")
PORTAINER_OPENID_CLIENT_ID = os.environ.get("PORTAINER_OPENID_CLIENT_ID")
PORTAINER_OPENID_CLIENT_SECRET = os.environ.get("PORTAINER_OPENID_CLIENT_SECRET")

REGISTRY_URL = f"registry.{PROJECT_BASE_URL}"

PORTAINER_API_URL = f"http://portainer:9000/api"
PORTAINER_TOKEN_VALID_DURATION = 8
PORTAINER_AUTH_METHOD_OAUTH_ID = 3

PLATFORM_TEAM_NAME = "platform-users"
PLATFORM_ENDPOINT_GROUP_NAME = "platform-edge-devices"

VERIFY_SSL = bool_from_env("BACKEND_VERIFY_SSL")


def _update_edge_key(endpoint: dict):
    # we need to set the SSH-Tunnel URL to the correct value
    # to work with Traefik

    # get the key and add extra padding if there is none
    # portainer strips the padding but python needs them
    # any additional == dont do any harm, so we can just add them all the time
    edge_key_encoded = endpoint["EdgeKey"] + "=="
    edge_key = base64.b64decode(edge_key_encoded).decode("utf-8")
    edge_key_tokens = edge_key.split("|")

    edge_key_tokens[1] = f"edge.{PROJECT_BASE_URL}:{PORTAINER_EDGE_PORT}"
    edge_key = "|".join(edge_key_tokens)

    # enocode the updated string back to base64 and remove the padding
    edge_key_encoded = (
        base64.b64encode(edge_key.encode("utf-8")).decode("utf-8").replace("=", "")
    )
    endpoint["EdgeKey"] = edge_key_encoded


def _error_message_from_response(response: requests.Response) -> str:
    """Builds an error message from a Portainer response.

    Docker answers a successful container removal with 204 and an empty body, and Portainer sends a
    JSON body with a 'message' only on error. Never rely on that body: a request for an edge device
    whose agent is not connected is answered with 502 and an empty body.
    """
    if response.content:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("message"):
            return str(body["message"])

    detail = response.text.strip()
    if detail:
        if len(detail) > 500:
            detail = detail[:500] + "..."
        return f"Portainer returned HTTP {response.status_code}: {detail}"

    return (
        f"Portainer returned HTTP {response.status_code} with an empty body "
        "(the agent of the edge device is probably not connected)"
    )


class DeploymentStartError(Exception):
    """The container was created on the device, but its start failed.

    ``container_id`` is only set if the container could not be removed again and
    therefore stays on the device.
    """

    def __init__(self, message: str, container_id: str = None):
        super().__init__(message)
        self.message = message
        self.container_id = container_id


class PortainerAPI:
    def __init__(self) -> None:
        self.jwt = None
        self.jwt_valid_until = None

        # Default to "Unassigned" Group (1)
        # This will be overridden by adding / fetching the endpoint group
        self.endpoint_group_id = 1

        self.team_id = None

        self._login_if_needed()

    def _is_logged_in(self) -> bool:
        if self.jwt is None:
            return False

        if self.jwt_valid_until < datetime.datetime.now():
            return False

        return True

    def _login_if_needed(self) -> bool:
        if self._is_logged_in():
            return True

        logger.info("Trying to log into Portainer!")
        try:
            response = requests.post(
                url=f"{PORTAINER_API_URL}/auth",
                json={
                    "username": os.environ.get("PORTAINER_ADMIN_USERNAME"),
                    "password": os.environ.get("PORTAINER_ADMIN_PASSWORD"),
                },
                verify=VERIFY_SSL,
            )
            response.raise_for_status()

            self.jwt = response.json()["jwt"]
            self.jwt_valid_until = datetime.datetime.now() + datetime.timedelta(
                hours=PORTAINER_TOKEN_VALID_DURATION
            )
            logger.info("Logged into Portainer!")
            return True
        except HTTPError as e:
            logger.error("Unable to Login to Portainer...")
            print(e)
            return False

    def _get_auth_header(self) -> Dict:
        return {
            "Authorization": f"Bearer {self.jwt}",
        }

    # do some initial setup after starting (for the first time)
    def setup(self):
        # setup the endpoint group that all edge devices will be added to
        self.endpoint_group_id = self._add_platform_endpoint_group()
        logger.debug(f"Set Portainer Endpoint Group ID to {self.endpoint_group_id}")

        # setup the team used by platform users
        self.team_id = self._add_platform_team()
        logger.debug(f"Set Portainer Team ID to {self.team_id}")

        # setup keycloak oauth
        self._add_keycloak()

        # setup platform-registry
        self._add_platform_registry()

    ### REGISTRY

    def _get_all_registries(self) -> dict:
        self._login_if_needed()

        response = requests.get(
            url=f"{PORTAINER_API_URL}/registries",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()

        return response.json()

    def _add_platform_registry(self) -> dict:
        registries = self._get_all_registries()
        if len(registries) > 0:
            logger.info("Platform Registry found in Portainer. Skipping!")
            return

        logger.info("Platform Registry not found in Portainer. Adding!")

        self._login_if_needed()

        response = requests.post(
            url=f"{PORTAINER_API_URL}/registries",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            json={
                "URL": REGISTRY_URL,
                "name": "platform-registry",
                "authentication": True,
                "username": REALM_SERVICE_ACCOUNT_USERNAME,
                "password": REALM_SERVICE_ACCOUNT_PASSWORD,
                "type": 3,
            },
        )
        response.raise_for_status()

        return response.json()

    ### ENDPOINT GROUPS - not to confuse with Edge Groups!

    def _get_endpoint_groups(self) -> dict:
        self._login_if_needed()
        response = requests.get(
            url=f"{PORTAINER_API_URL}/endpoint_groups",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()
        return response.json()

    def _add_platform_endpoint_group(self) -> int:
        groups = self._get_endpoint_groups()
        platform_group = next(
            (g for g in groups if g["Name"] == PLATFORM_ENDPOINT_GROUP_NAME), None
        )
        if platform_group is not None:
            logger.info("Platform Group with found in Portainer. Skipping!")
            return platform_group["Id"]

        logger.info("Platform Group not found in Portainer. Adding!")

        self._login_if_needed()
        response = requests.post(
            url=f"{PORTAINER_API_URL}/endpoint_groups",
            json={
                "Name": PLATFORM_ENDPOINT_GROUP_NAME,
                "AssociatedEndpoints": [],
                "Description": "The Group that is used by all edge devices added by the Platform",
                "TagIds": [],
            },
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()
        return response.json()["Id"]

    ### TEAM

    def _get_teams(self) -> dict:
        self._login_if_needed()
        response = requests.get(
            url=f"{PORTAINER_API_URL}/teams",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()
        return response.json()

    def _add_platform_team(self) -> int:
        # check if the team already exists
        teams = self._get_teams()
        platform_team = next(
            (t for t in teams if t["Name"] == PLATFORM_TEAM_NAME), None
        )
        if platform_team is not None:
            logger.info("Platform Team found in Portainer. Skipping!")
            return platform_team["Id"]

        logger.info("Platform Team not found in Portainer. Adding!")

        self._login_if_needed()
        # create the team
        response = requests.post(
            url=f"{PORTAINER_API_URL}/teams",
            json={"Name": PLATFORM_TEAM_NAME},
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()

        team_id = response.json()["Id"]

        # give the team access to the edge device group
        response = requests.put(
            url=f"{PORTAINER_API_URL}/endpoint_groups/{self.endpoint_group_id}",
            json={"TeamAccessPolicies": {team_id: {"RoleId": 0}}},
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()

        # return the team id
        return team_id

    ### OAUTH

    def _add_keycloak(self):
        """
        Configuration Setup taken from: https://www.youtube.com/watch?v=5KaTFPoGo3I
        """

        self._login_if_needed()

        response = requests.get(
            url=f"{PORTAINER_API_URL}/settings",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()
        settings = response.json()

        if settings["AuthenticationMethod"] == PORTAINER_AUTH_METHOD_OAUTH_ID:
            logger.info("Portainer OAuth already configured! Skipping...")
            return

        logger.info("Setting up Portainer OAuth Client!")

        oauth_settings = {
            "ClientID": PORTAINER_OPENID_CLIENT_ID,
            "ClientSecret": PORTAINER_OPENID_CLIENT_SECRET,
            "AccessTokenURI": f"https://keycloak.{PROJECT_BASE_URL}/realms/test-realm/protocol/openid-connect/token",
            "AuthorizationURI": f"https://keycloak.{PROJECT_BASE_URL}/realms/test-realm/protocol/openid-connect/auth",
            "ResourceURI": f"https://keycloak.{PROJECT_BASE_URL}/realms/test-realm/protocol/openid-connect/userinfo",
            "RedirectURI": f"https://portainer.{PROJECT_BASE_URL}/",
            "DefaultTeamID": self.team_id,
            "LogoutURI": "",
            "OAuthAutoCreateUsers": True,
            "SSO": True,
            "Scopes": "openid",
            "UserIdentifier": "preferred_username",
        }

        response = requests.put(
            url=f"{PORTAINER_API_URL}/settings",
            json={
                "AuthenticationMethod": PORTAINER_AUTH_METHOD_OAUTH_ID,
                "OAuthSettings": oauth_settings,
            },
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()

    ### ENDPOINTS

    def get_all_endpoints(self) -> dict:
        self._login_if_needed()
        try:
            response = requests.get(
                url=f"{PORTAINER_API_URL}/endpoints",
                verify=VERIFY_SSL,
                headers=self._get_auth_header(),
            )
            response.raise_for_status()

            endpoints_list = response.json()
            endpoints = dict()
            for e in endpoints_list:
                endpoints[e["Name"]] = e

            return endpoints
        except HTTPError as e:
            logger.error("Unable to fetch Portainer Endpoints...")
            return []

    def create_new_endpoint(self, name: str, tag_ids: List[int]) -> dict:
        self._login_if_needed()
        response = requests.post(
            url=f"{PORTAINER_API_URL}/endpoints",
            params={
                "Name": name,
                "EndpointCreationType": 4,  # Edge Agent
                "URL": f"https://portainer.{PROJECT_BASE_URL}",
                "GroupID": self.endpoint_group_id,
                "TagIds": json.dumps(tag_ids),
            },
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()

        endpoint = response.json()
        logger.debug(endpoint)

        _update_edge_key(endpoint)

        return endpoint

    def delete_endpoint(self, endpoint_id: int):
        self._login_if_needed()
        response = requests.delete(
            url=f"{PORTAINER_API_URL}/endpoints/{endpoint_id}",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()

    def get_endpoint_docker_info(self, endpoint_id: int) -> dict:
        self._login_if_needed()
        response = requests.get(
            url=f"{PORTAINER_API_URL}/endpoints/{endpoint_id}/docker/info",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )
        response.raise_for_status()

        return response.json()

    ### DEPLOYMENT

    def check_if_deployment_exists(
        self, endpoint_id: int, deployment_name: str
    ) -> bool:
        self._login_if_needed()

        # request an existing container that matches the name of the new container
        # if there is a match, we can not deploy another container with the same name
        response = requests.get(
            url=f"{PORTAINER_API_URL}/endpoints/{endpoint_id}/docker/containers/json",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            params={
                "all": 1,
                "filters": json.dumps({"name": [f"^/{deployment_name}$"]}),
            },
        )

        if not response.ok:
            response.raise_for_status()

        deployments = response.json()

        return len(deployments) != 0

    def deploy_container_to_edge_device(
        self,
        edge_device: EdgeDevice,
        container_image: ContainerImage,
        container_deployment: ContainerDeployment,
    ):
        self._login_if_needed()

        auth_header = self._get_auth_header()
        image = f"{REGISTRY_URL}/{container_image.repository}:{container_image.tag}"

        # we define which registry to use
        # here we use the registry setup in portainer as id 1, which is the local platform registry
        # this MIGHT break IF somehow a user adds another registry manually, BEFORE the backend started
        # if this might ever happen: fetch all registries and look for the one matching the REGISTRY_URL...
        registry_credentials_json = json.dumps({"registryId": 1})
        registry_auth = base64.urlsafe_b64encode(
            registry_credentials_json.encode("UTF-8")
        ).decode("UTF-8")

        # pull
        response = requests.post(
            url=f"{PORTAINER_API_URL}/endpoints/{edge_device.portainer_id}/docker/images/create",
            verify=VERIFY_SSL,
            headers={
                "Authorization": auth_header["Authorization"],
                "X-Registry-Auth": registry_auth,
                "Content-Type": "application/json",
            },
            params={"fromImage": image},
        )
        if not response.ok:
            raise HTTPError(response.status_code, str(response.text))

        # prepare port bindings:
        exposed_ports = dict()
        host_port_bindings = dict()
        for pb in container_deployment.port_bindings:
            exposed_ports[f"{pb.container_port}/{pb.protocol.value}"] = {}
            host_port_bindings[f"{pb.container_port}/{pb.protocol.value}"] = [
                {"HostPort": str(pb.host_port)}
            ]

        # create
        response = requests.post(
            url=f"{PORTAINER_API_URL}/endpoints/{edge_device.portainer_id}/docker/containers/create",
            verify=VERIFY_SSL,
            headers=auth_header,
            params={"name": container_deployment.name},
            json={
                "Image": image,
                "ExposedPorts": exposed_ports,
                "HostConfig": {"AutoRemove": False, "PortBindings": host_port_bindings, "RestartPolicy": { "Name": "unless-stopped", "MaximumRetryCount": 0 } },
            },
        )
        if not response.ok:
            msg = response.json()["message"]
            logger.debug(msg)
            raise HTTPException(400, msg)

        create_container_response = response.json()
        container_id = create_container_response["Id"]

        try:
            # get resource control ID of container just created
            rc_id = create_container_response["Portainer"]["ResourceControl"]["Id"]

            # extract already-set user and team IDs
            user_ids = [
                user.get("UserId")
                for user in create_container_response["Portainer"]["ResourceControl"]["UserAccesses"]
            ]
            team_ids = [
                team.get("TeamId")
                for team in create_container_response["Portainer"]["ResourceControl"]["TeamAccesses"]
            ]
            # append team ID of platform_users
            team_ids.append(self.team_id)

            # update resource control to allow access from platform_users
            rc_response = requests.put(
                url=f"{PORTAINER_API_URL}/resource_controls/{rc_id}",
                verify=VERIFY_SSL,
                headers=auth_header,
                json={
                    "administratorsOnly": False,
                    "public": False,
                    "teams": team_ids,
                    "users": user_ids
                },
            )
            if not rc_response.ok:
                logger.error(rc_response)
                raise HTTPException(
                    400, _error_message_from_response(rc_response)
                )
        
            # start
            response = requests.post(
                url=f"{PORTAINER_API_URL}/endpoints/{edge_device.portainer_id}/docker/containers/{container_id}/start",
                verify=VERIFY_SSL,
                headers=auth_header,
            )
            if not response.ok:
                msg = _error_message_from_response(response)
                logger.debug(msg)
                raise HTTPException(400, msg)
        except Exception as e:
            message = e.detail if isinstance(e, HTTPException) else str(e)
            message = str(message)
            logger.error("Start of container %s failed: %s", container_id, message)
            # a failed start must not leave the container (and its name) on the device
            try:
                self._remove_container(edge_device.portainer_id, container_id)
            except Exception as cleanup_error:
                raise DeploymentStartError(
                    f"{message}; the container {container_id[:12]} could not be "
                    f"removed ({cleanup_error}) and stays on the device",
                    container_id=container_id,
                ) from e
            raise DeploymentStartError(
                f"{message}; the container was removed from the device again"
            ) from e

        return container_id

    def _remove_container(self, endpoint_id: int, container_id: str) -> None:
        """Removes a container (force, with its anonymous volumes). A missing container counts as removed."""
        self._login_if_needed()
        response = requests.delete(
            url=f"{PORTAINER_API_URL}/endpoints/{endpoint_id}/docker/containers/{container_id}",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            params={"v": 1, "force": True},
            timeout=30,
        )
        if not (response.ok or response.status_code == 404):
            raise HTTPError(_error_message_from_response(response))

    def list_containers(self, endpoint_id: int) -> List[dict]:
        """Lists all containers of an endpoint as name, state and published host ports."""
        self._login_if_needed()
        response = requests.get(
            url=f"{PORTAINER_API_URL}/endpoints/{endpoint_id}/docker/containers/json",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            params={"all": 1},
            timeout=15,
        )
        if not response.ok:
            raise HTTPError(_error_message_from_response(response))

        containers = []
        for c in response.json():
            names = c.get("Names") or []
            name = names[0].lstrip("/") if names else ""
            ports = sorted(
                {
                    p["PublicPort"]
                    for p in (c.get("Ports") or [])
                    if p.get("PublicPort")
                }
            )
            containers.append(
                {"name": name, "state": c.get("State") or "", "host_ports": ports}
            )
        return containers

    def undeploy_container_from_edge_device(
        self, edge_device: EdgeDevice, container_deployment: ContainerDeployment
    ):
        self._login_if_needed()

        response = requests.delete(
            url=f"{PORTAINER_API_URL}/endpoints/{edge_device.portainer_id}/docker/containers/{container_deployment.docker_container_id}",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            params={"v": 1, "force": True},
        )

        # if we try to delete a non existing container deployment, it was likely deleted manually (outside of the platform)
        # just ignore this case and return True
        # a successful removal is 204 with an empty body, so the status alone decides about success
        if response.ok or response.status_code == requests.codes.not_found:
            return True

        msg = _error_message_from_response(response)
        logger.error(msg)
        raise HTTPException(400, msg)

    ### TAGS
    def get_tags(self):
        self._login_if_needed()

        response = requests.get(
            url=f"{PORTAINER_API_URL}/tags",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )

        if not response.ok:
            response.raise_for_status()

        tags = response.json()

        return tags

    def create_tag(self, name: str):
        self._login_if_needed()

        response = requests.post(
            url=f"{PORTAINER_API_URL}/tags",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            json={"name": name},
        )

        if not response.ok:
            response.raise_for_status()

        tag = response.json()

        return tag

    def get_ids_for_tag_names(
        self, tag_names: List[str], allow_create=False
    ) -> List[int]:
        # check if a tag with the given name exists and put it's id in the list
        # if not, create it and then add it's id to the list.

        tag_ids = []
        self._login_if_needed()

        # convert portainer object to dict
        existing_tags = self.get_tags()
        existing_tag_dict = dict()
        for t in existing_tags:
            existing_tag_dict[t["Name"]] = t["ID"]

        not_existing_tags = []

        for tag in tag_names:
            if tag in existing_tag_dict.keys():
                # requested tag exists
                tag_ids.append(existing_tag_dict[tag])
            else:
                # tag does not exist, add to list to create:
                not_existing_tags.append(tag)

        # create all not existing tags:
        if allow_create:
            for tag in not_existing_tags:
                new_tag = self.create_tag(tag)
                tag_ids.append(new_tag["ID"])

        return tag_ids

    ### EDGE GROUPS
    # Groups that are dynamically made up off tags

    def get_all_edge_groups(self):
        self._login_if_needed()

        response = requests.get(
            url=f"{PORTAINER_API_URL}/edge_groups",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )

        if not response.ok:
            response.raise_for_status()

        groups = response.json()

        return groups

    def create_edge_group(self, name: str, tag_ids: List[int]):
        self._login_if_needed()

        response = requests.post(
            url=f"{PORTAINER_API_URL}/edge_groups",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            json={
                "Name": name,
                "Endpoints": [],
                "Dynamic": True,
                "TagIds": tag_ids,
                "PartialMatch": False,
            },
        )

        if not response.ok:
            response.raise_for_status()

        group = response.json()

        return group

    def delete_edge_group(self, edge_group_id: int):
        self._login_if_needed()

        response = requests.delete(
            url=f"{PORTAINER_API_URL}/edge_groups/{edge_group_id}",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )

        if not response.ok:
            response.raise_for_status()

        return True

    ### EDGE STACK
    def get_all_edge_stacks(self):
        self._login_if_needed()

        response = requests.get(
            url=f"{PORTAINER_API_URL}/edge_stacks",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )

        if not response.ok:
            response.raise_for_status()

        stacks = response.json()

        return stacks

    def get_edge_stack(self, edge_stack_id: int):
        self._login_if_needed()

        response = requests.get(
            url=f"{PORTAINER_API_URL}/edge_stacks/{edge_stack_id}",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )

        if not response.ok:
            response.raise_for_status()

        stack = response.json()

        return stack

    def deploy_edge_stack(self, name: str, edge_group_ids: List[int], yaml: str):
        self._login_if_needed()

        response = requests.post(
            url=f"{PORTAINER_API_URL}/edge_stacks",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            params={"method": "string"},
            json={
                "name": name,
                "DeploymentType": 0,
                "StackFileContent": yaml,
                "EdgeGroups": edge_group_ids,
            },
        )

        if not response.ok:
            response.raise_for_status()

        stack = response.json()

        return stack

    def edit_edge_stack(self, edge_stack_id: int, edge_group_ids: List[int], yaml: str):
        self._login_if_needed()

        current_stack = self.get_edge_stack(edge_stack_id)
        current_version = current_stack["Version"]
        new_version = current_version + 1

        response = requests.put(
            url=f"{PORTAINER_API_URL}/edge_stacks/{edge_stack_id}",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
            json={
                "StackFileContent": yaml,
                "DeploymentType": 0,
                "EdgeGroups": edge_group_ids,
                "Version": new_version,
            },
        )

        if not response.ok:
            response.raise_for_status()

        stack = response.json()

        return stack

    def delete_edge_stack(self, edge_stack_id: int):
        self._login_if_needed()

        response = requests.delete(
            url=f"{PORTAINER_API_URL}/edge_stacks/{edge_stack_id}",
            verify=VERIFY_SSL,
            headers=self._get_auth_header(),
        )

        if not response.ok:
            response.raise_for_status()

        return True


portainer = PortainerAPI()
