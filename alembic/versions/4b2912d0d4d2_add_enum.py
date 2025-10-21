"""add enum

Revision ID: 4b2912d0d4d2
Revises: 
Create Date: 2025-10-20 14:35:40.834825

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4b2912d0d4d2'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Сначала создаем enum тип в PostgreSQL
    op.execute("""
        CREATE TYPE authoritytype AS ENUM (
            'PRESIDENT', 'GOVERNMENT', 'STATE', 'MVD', 'CIVIL_DEFENSE', 'MID', 'SNG', 'MO', 
            'SVTS', 'STEK', 'ASS', 'MU', 'SIN', 'SSP', 'OK', 'SDK', 'AVC', 'PAK', 'FDS', 
            'FSB', 'FSVNG', 'FSKON', 'FSO', 'SVR', 'FSFM', 'GYSPP', 'YDP', 'MZ', 'FSNSZ', 
            'FMBA', 'MK', 'FAA', 'FAT', 'MNVO', 'MON', 'SNSON', 'FADM', 'MPRE', 'SGMOS', 
            'SNP', 'AVR', 'ALH', 'AN', 'MPT', 'MP', 'ATRM', 'MRDVA', 'MCRSMK', 'MSMK', 
            'SNSSITMK', 'APMK', 'AG', 'MSH', 'SVFN', 'FAR', 'MS', 'MSJKH', 'FPP', 'MT', 
            'FSNST', 'FAVT', 'FDA', 'FAJT', 'FAMRT', 'MTSZ', 'FSRZ', 'MF', 'FNS', 'FSFBN', 
            'FK', 'MER', 'FSA', 'FSGRKK', 'FSIS', 'FAGR', 'FAYGI', 'ME', 'FAS', 'FSGS', 
            'FMS', 'FSNZPP', 'GGSV', 'FSRAR', 'FSKATR', 'FTS', 'FST', 'FSETAN', 'FKA', 
            'FANO', 'FADN', 'FAOGG', 'FSS', 'FFOMS', 'PF', 'FPSS', 'PPF', 'SP', 'SK', 
            'CB', 'GP', 'GPR', 'RAN'
        )
    """)

    # Добавляем колонку как nullable=True
    op.add_column('data_legislation',
                  sa.Column('authority',
                            sa.Enum('PRESIDENT', 'GOVERNMENT', 'STATE', 'MVD', 'CIVIL_DEFENSE', 'MID', 'SNG', 'MO',
                                    'SVTS', 'STEK', 'ASS', 'MU', 'SIN', 'SSP', 'OK', 'SDK', 'AVC', 'PAK', 'FDS', 'FSB',
                                    'FSVNG', 'FSKON', 'FSO', 'SVR', 'FSFM', 'GYSPP', 'YDP', 'MZ', 'FSNSZ', 'FMBA', 'MK',
                                    'FAA', 'FAT', 'MNVO', 'MON', 'SNSON', 'FADM', 'MPRE', 'SGMOS', 'SNP', 'AVR', 'ALH',
                                    'AN', 'MPT', 'MP', 'ATRM', 'MRDVA', 'MCRSMK', 'MSMK', 'SNSSITMK', 'APMK', 'AG',
                                    'MSH', 'SVFN', 'FAR', 'MS', 'MSJKH', 'FPP', 'MT', 'FSNST', 'FAVT', 'FDA', 'FAJT',
                                    'FAMRT', 'MTSZ', 'FSRZ', 'MF', 'FNS', 'FSFBN', 'FK', 'MER', 'FSA', 'FSGRKK', 'FSIS',
                                    'FAGR', 'FAYGI', 'ME', 'FAS', 'FSGS', 'FMS', 'FSNZPP', 'GGSV', 'FSRAR', 'FSKATR',
                                    'FTS', 'FST', 'FSETAN', 'FKA', 'FANO', 'FADN', 'FAOGG', 'FSS', 'FFOMS', 'PF',
                                    'FPSS', 'PPF', 'SP', 'SK', 'CB', 'GP', 'GPR', 'RAN',
                                    name='authoritytype',
                                    create_constraint=False),
                            nullable=True)
                  )

    # Обновляем существующие строки - устанавливаем значение 'PRESIDENT' для всех
    data_legislation = sa.table('data_legislation',
                             sa.column('authority')
                             )
    op.execute(
        data_legislation.update().values(authority='PRESIDENT')
    )

    # Теперь меняем колонку на NOT NULL
    op.alter_column('data_legislation', 'authority', nullable=False)

    # Создаем индекс
    op.create_index(op.f('ix_data_legislation_authority'), 'data_legislation', ['authority'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Удаляем индекс
    op.drop_index(op.f('ix_data_legislation_authority'), table_name='data_legislation')

    # Удаляем колонку
    op.drop_column('data_legislation', 'authority')

    # Удаляем enum тип
    op.execute("DROP TYPE authoritytype")
