# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Adding config_override to TrainContainer.

Revision ID: 7d3a1c5e9b42
Revises: afd44e1e43ee
Create Date: 2026-10-20 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "7d3a1c5e9b42"
down_revision = "afd44e1e43ee"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "train_containers", sa.Column("config_override", sa.JSON(), nullable=True)
    )


def downgrade():
    op.drop_column("train_containers", "config_override")
