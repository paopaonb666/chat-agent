"""add path column to knowledge_documents

Revision ID: 648c36d877fd
Revises: 9d558fd77b2f
Create Date: 2026-05-26 11:00:25.973549

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '648c36d877fd'
down_revision: Union[str, Sequence[str], None] = '9d558fd77b2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('knowledge_documents', sa.Column('path', sa.String(), nullable=False, server_default=''))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('knowledge_documents', 'path')
