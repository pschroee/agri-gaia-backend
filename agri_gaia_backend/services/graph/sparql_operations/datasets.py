# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
# SPDX-FileContributor: Henri Graf
# SPDX-FileContributor: Jonas Tüpker
# SPDX-FileContributor: Lukas Hesse
# SPDX-FileContributor: Maik Fruhner
# SPDX-FileContributor: Prof. Dr.-Ing. Heiko Tapken
# SPDX-FileContributor: Tobias Wamhof
#
# SPDX-License-Identifier: MIT

import os

from rdflib.namespace import RDF, DCTERMS, XSD, PROV
from rdflib import Namespace, Graph, Literal, URIRef, OWL, BNode, DCAT

from . import util

import logging
import json
import yaml

FUSEKI_ENDPOINT = os.environ.get("FUSEKI_ENDPOINT")
SPARQL_UPDATE_ENDPOINT = f"{FUSEKI_ENDPOINT}ds/update"
SPARQL_QUERY_ENDPOINT = f"{FUSEKI_ENDPOINT}ds/sparql"

logger = logging.getLogger("api-logger")


def query_for_concepts(concept_uris):
    """Queries for all datasets, which are annotated by the one of the defined concepts.

    Args:
        concept_uris: A list including all concept uris, which should be used for the search.

    Returns:
        A list of all dataset URIs in the RDF storage, which are annotated by one of the given concepts.
    """
    possible_classes = util.query_possible_data_ressources()
    final_classes = []
    for possible_class in possible_classes:
        possible_class = "<" + possible_class + ">"
        final_classes.append(possible_class)
    classes_string = ",".join(final_classes)
    for concept in concept_uris:
        concept = "<" + concept + ">"
    concept_string = ",".join(concept_uris)
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?iri WHERE {{
            ?iri a ?type .
            filter (?type IN ({classes_string}))
            ?iri <http://www.w3.org/ns/dcat#keyword> ?obj .
            filter (?obj IN ({concept_string}))
        }}"""

    print(query)

    response = util.send_query(SPARQL_QUERY_ENDPOINT, query)
    uris = []
    if len(response) > 0:
        for res in response["results"]["bindings"]:
            uris.append(res["iri"]["value"])
    return uris


def get_labels_for_dataset(dataset_id):
    query = f"""Select ?obj WHERE {{
        <{dataset_id}> <http://www.w3.org/ns/dcat#keyword> ?obj
    }}"""

    response = util.send_query(SPARQL_QUERY_ENDPOINT, query)["results"]["bindings"]
    labels = []
    if len(response) > 0:
        for entry in response:
            labels.append(entry["obj"]["value"])
    return labels


def get_description_for_dataset(dataset_id):
    query = f"""Select ?obj WHERE {{
        <{dataset_id}> <http://www.w3.org/ns/dcat#description> ?obj
    }}"""

    response = util.send_query(SPARQL_QUERY_ENDPOINT, query)["results"]["bindings"]
    description = ""
    if len(response) > 0:
        description = response[0]["obj"]["value"]
    return description


def get_metadata_information(dataset_id):
    query = f"""Select ?sub ?pred ?obj WHERE {{
        ?sub ?pred ?obj
        Filter(?sub = <{dataset_id}>)
    }}"""

    response = util.send_query(SPARQL_QUERY_ENDPOINT, query)["results"]["bindings"]
    return response


AGRI_GAIA_ASSET_NAMESPACE = "http://w3id.org/agri-gaia-x/asset#"


def get_user_metadata_for_uri(uri: str):
    """Returns the metadata the uploader passed at creation as key/value pairs.

    _handle_type stores it as properties in the agri-gaia asset namespace next to the
    standard DCAT triples. The experiment runner compares these values with its fixture
    manifest (X4, #146); the rest of the graph is not part of that metadata.
    """
    query = f"""SELECT ?p ?o WHERE {{
        <{uri}> ?p ?o .
        FILTER(STRSTARTS(STR(?p), "{AGRI_GAIA_ASSET_NAMESPACE}"))
    }}"""
    response = util.send_query(SPARQL_QUERY_ENDPOINT, query)["results"]["bindings"]
    metadata = {}
    for row in response:
        key = row["p"]["value"].rsplit("#", 1)[-1]
        metadata[key] = _binding_value(row["o"])
    return metadata


def _binding_value(binding):
    """Returns a SPARQL binding as the JSON value the metadata came from.

    The graph stores the values as literals; typed literals are converted back, so that a
    number or a boolean in the manifest can be compared with the stored value.
    """
    value = binding.get("value")
    datatype = binding.get("datatype", "")
    if datatype.endswith("#boolean"):
        return value == "true"
    if datatype.endswith(("#integer", "#int", "#long", "#short", "#byte")):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if datatype.endswith(("#double", "#float", "#decimal")):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def delete_dataset(dataset_id):
    dataset_id_splitted = dataset_id.split("#")
    dataset_id_splitted[-1] = "_Distribution"
    distribution_id = "#".join(dataset_id_splitted)

    query = f"""
        DELETE{{  ?id ?p ?o }}
            WHERE{{   
                {{
                    FILTER (?id = <{dataset_id}>)
                    ?id ?p ?o
                }}
                UNION
                {{
                    FILTER (?id = <{distribution_id}>)
                    ?id ?p ?o
                }}
                
            }}"""

    return util.send_update(SPARQL_UPDATE_ENDPOINT, query)


def get_default_graph(minio_server, bucket, dataset_name, dataset_id):
    """
    Returns a default graph for a dataset, that is located inside a bucket on a server located at the passed url

    Args:
        minio_server: The location of the MinIO instance.
        bucket: the bucket, where the dataset is stored. Is identical with the creator username.
        dataset_name: The dataset name.
        dataset_id: The dataset ID.

    Returns:
        The default graph.
    """
    graph = Graph()
    dcat = Namespace("http://www.w3.org/ns/dcat#")

    graph.bind("dcat", dcat)
    graph.bind("dct", DCTERMS)
    id_string = (
        "https://" + minio_server + "/" + bucket + "/datasets/" + str(dataset_id)
    )
    accessURL = URIRef(id_string)
    datasetid = URIRef(id_string + "#_Dataset")
    distributionid = URIRef(id_string + "#_Distribution")
    graph.add((datasetid, dcat.title, Literal(dataset_name)))
    graph.add((datasetid, dcat.publisher, Literal(bucket)))
    graph.add((datasetid, dcat.distribution, distributionid))
    graph.add((distributionid, dcat.accessURL, accessURL))
    return graph, id_string


def create_graph(
    label_uris,
    location_uris,
    minio_server: str,
    bucket: str,
    dataset_name: str,
    dataset_id: int,
    description: str,
    metadata: str,
    dataset_type: str,
    config_files,
    annotation_labels,
):
    """
    Returns a graph for a dataset, that is located inside a bucket on a server located at the passed url.
    The dataset has to be annotated with information on used labels, the creator and the creation date.

    Args:
        label_uris: The URIs of the keywords, describing the dataset.
        location_uris: The URIs of the locations, describing the dataset.
        minio_server: The location of the MinIO Instance.
        bucket: The bucket, where the dataset is uploaded.
        dataset_name: the Dataset name.
        dataset_id: The Dataset ID.
        description: The dataset description.
        metadata: Additional metadata for the dataset.
        dataset_type: The dataset type.
        config_files: A dictionary containing configuration files. They can be used to extract additional metadata.

    Returns:
        A graph object for the dataset.
    """
    (graph, id_string) = get_default_graph(
        minio_server, bucket, dataset_name, dataset_id
    )
    dcat = Namespace("http://www.w3.org/ns/dcat#")

    datasetid = URIRef(id_string + "#_Dataset")
    if annotation_labels is not None:
        for label in annotation_labels:
            graph.add((datasetid, dcat.keyword, Literal(label)))
    for uri in label_uris:
        graph.add((datasetid, dcat.keyword, uri))
    for uri in location_uris:
        graph.add((datasetid, DCTERMS.spatial, uri))
    graph.add((datasetid, dcat.description, Literal(description)))

    graph = _handle_type(
        graph=graph,
        metadata=metadata,
        dataset_id=datasetid,
        dataset_type=dataset_type,
        config_files=config_files,
    )

    return graph, datasetid


def _handle_type(graph, metadata, dataset_id: URIRef, dataset_type: str, config_files):
    metadata = json.loads(metadata)

    if dataset_type == "AgriSyntheticImageDataResource":
        graph, metadata = _handle_synthetic_dataset(
            graph=graph,
            metadata=metadata,
            config_files=config_files,
            fuseki_id=dataset_id,
        )

    agri_gaia = Namespace("http://w3id.org/agri-gaia-x/asset#")

    graph.bind("agri_gaia", agri_gaia)

    graph.add((dataset_id, RDF.type, getattr(agri_gaia, dataset_type)))
    for key, value in metadata.items():
        graph.add((dataset_id, getattr(agri_gaia, key), Literal(value)))

    return graph


def _handle_synthetic_dataset(graph, metadata, config_files, fuseki_id):
    if (
        config_files
        and config_files["config"] is not None
        and config_files["asset_catalog"] is not None
    ):
        config_file_yaml = yaml.safe_load(config_files["config"].file.read())
        asset_catalog_file_yaml = yaml.safe_load(
            config_files["asset_catalog"].file.read()
        )

        graph.bind("agri-gax", "http://w3id.org/agri-gaia-x/asset#")
        graph.bind("gax-trust-framework", "http://w3id.org/gaia-x/gax-trust-framework#")
        graph.bind("hydra", "http://www.w3.org/ns/hydra/core#")
        graph.bind("schema", "http://schema.org/")
        graph.bind("xsd", XSD)
        graph.bind("prov", PROV)
        graph.bind("owl", OWL)

        if config_file_yaml["steps"]:
            graph.add(
                (
                    fuseki_id,
                    URIRef("http://w3id.org/agri-gaia-x/asset#imageCount"),
                    Literal(config_file_yaml["steps"]),
                )
            )

        if config_file_yaml["postprocessing"]["Postprocessing/BoundingBoxes"][0]["id"]:
            graph.add(
                (
                    fuseki_id,
                    URIRef("http://w3id.org/agri-gaia-x/asset#boundingBoxFormat"),
                    Literal(
                        config_file_yaml["postprocessing"][
                            "Postprocessing/BoundingBoxes"
                        ][0]["id"],
                        datatype=XSD.string,
                    ),
                )
            )

        base_camera_plugins = config_file_yaml["sensor"]["Base Plugins/Camera"][0]
        if base_camera_plugins["name"]:
            bnode_device = BNode()
            graph.add(
                (
                    fuseki_id,
                    URIRef("http://w3id.org/agri-gaia-x/asset#deviceUsed"),
                    bnode_device,
                )
            )
            graph.add(
                (
                    bnode_device,
                    RDF.type,
                    URIRef("http://w3id.org/agri-gaia-x/asset#VirtualCamera"),
                )
            )

            if base_camera_plugins["sensor_width"]:
                graph.add(
                    (
                        bnode_device,
                        URIRef("http://w3id.org/agri-gaia-x/asset#cameraSensor"),
                        Literal(
                            str(base_camera_plugins["sensor_width"]) + " mm",
                            datatype=XSD.string,
                        ),
                    )
                )

            if base_camera_plugins["focal_length"]:
                graph.add(
                    (
                        bnode_device,
                        URIRef("http://w3id.org/agri-gaia-x/asset#cameraLense"),
                        Literal(
                            "Focal length "
                            + str(base_camera_plugins["focal_length"])
                            + " mm",
                            datatype=XSD.string,
                        ),
                    )
                )

            if base_camera_plugins["motion_blur"]["rolling_shutter"]["enabled"]:
                graph.add(
                    (
                        bnode_device,
                        URIRef("http://w3id.org/agri-gaia-x/asset#cameraShutter"),
                        Literal("Rolling shutter enabled", datatype=XSD.string),
                    )
                )

            if base_camera_plugins["resolution"]:
                graph.add(
                    (
                        bnode_device,
                        URIRef("http://w3id.org/agri-gaia-x/asset#cameraImageWidth"),
                        Literal(base_camera_plugins["resolution"][0]),
                    )
                )
                graph.add(
                    (
                        bnode_device,
                        URIRef("http://w3id.org/agri-gaia-x/asset#cameraImageHeight"),
                        Literal(base_camera_plugins["resolution"][1]),
                    )
                )
                graph.add(
                    (
                        fuseki_id,
                        URIRef("http://w3id.org/agri-gaia-x/asset#avgImageWidth"),
                        Literal(base_camera_plugins["resolution"][0]),
                    )
                )
                graph.add(
                    (
                        fuseki_id,
                        URIRef("http://w3id.org/agri-gaia-x/asset#avgImageHeight"),
                        Literal(base_camera_plugins["resolution"][1]),
                    )
                )

            if base_camera_plugins["shutter_speed"]:
                graph.add(
                    (
                        fuseki_id,
                        URIRef("http://w3id.org/agri-gaia-x/asset#imageExposure"),
                        Literal(
                            "Shutter speed "
                            + str(base_camera_plugins["shutter_speed"])
                            + " sec",
                            datatype=XSD.string,
                        ),
                    )
                )

            if base_camera_plugins["depth_of_field"]["aperture"]:
                img_exposure = "Aperature f/" + str(
                    base_camera_plugins["depth_of_field"]["aperture"]
                )
                if base_camera_plugins["depth_of_field"]["autofocus"]:
                    img_exposure = img_exposure + ", autofocus enabled"
                graph.add(
                    (
                        fuseki_id,
                        URIRef("http://w3id.org/agri-gaia-x/asset#imageExposure"),
                        Literal(img_exposure, datatype=XSD.string),
                    )
                )

            if base_camera_plugins["outputs"]["Base Plugins/RGB"]:
                graph.add(
                    (
                        fuseki_id,
                        URIRef("http://w3id.org/agri-gaia-x/asset#imageColorScheme"),
                        Literal("RGB", datatype=XSD.string),
                    )
                )

            if base_camera_plugins["outputs"]["Base Plugins/PixelAnnotation"]:
                for item in base_camera_plugins["outputs"][
                    "Base Plugins/PixelAnnotation"
                ][0]:
                    graph.add(
                        (
                            fuseki_id,
                            URIRef(
                                "http://w3id.org/agri-gaia-x/asset#availablePixelAnnotation"
                            ),
                            Literal(item, datatype=XSD.string),
                        )
                    )

        asset_index = []
        for item in config_file_yaml["scene"]:
            for scene_item in config_file_yaml["scene"][item]:
                for base_plugin_item in scene_item:
                    if base_plugin_item in ("texture", "models"):
                        if type(scene_item[base_plugin_item]) is list:
                            asset_index.append(
                                str(scene_item[base_plugin_item][0]).split("/")
                            )
                        elif type(base_plugin_item) is str:
                            asset_index.append(scene_item[base_plugin_item].split("/"))
                    if base_plugin_item == "environment_image":
                        if "random_selection" in scene_item["environment_image"]:
                            asset_index.append(
                                scene_item["environment_image"]["random_selection"][
                                    0
                                ].split("/")
                            )

        for item in asset_index:
            graph.add((fuseki_id, DCAT.keyword, Literal(item[1])))
            if "agrovoc_uri" in asset_catalog_file_yaml[item[0]]["assets"][item[1]]:
                graph.add(
                    (
                        fuseki_id,
                        DCTERMS.subject,
                        URIRef(
                            asset_catalog_file_yaml[item[0]]["assets"][item[1]][
                                "agrovoc_uri"
                            ]
                        ),
                    )
                )
                graph.add(
                    (
                        fuseki_id,
                        URIRef("http://w3id.org/agri-gaia-x/asset#objectsInTheScene"),
                        URIRef(
                            asset_catalog_file_yaml[item[0]]["assets"][item[1]][
                                "agrovoc_uri"
                            ]
                        ),
                    )
                )

    return graph, metadata
