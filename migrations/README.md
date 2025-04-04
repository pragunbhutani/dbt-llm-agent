# Database Migrations with Alembic

This directory contains database migration scripts using Alembic. Alembic is a database migration tool for SQLAlchemy that provides a way to incrementally update database schemas.

## How to Use

### Running Migrations

To run migrations and update your database to the latest schema version:

```bash
# First make sure your .env file has POSTGRES_URI set correctly
# Then run the migrate command
ragstar migrate

# Migrate to a specific revision
ragstar migrate --revision "revision_id"
```

### Initializing a New Database

To initialize a new database with the current schema:

```bash
# Make sure your .env file has POSTGRES_URI set correctly
ragstar init-db
```

### Creating New Migrations

To create a new migration script:

```bash
# Generate a new empty migration script
poetry run alembic revision -m "Description of changes"

# Auto-generate a migration script by comparing models to DB
poetry run alembic revision --autogenerate -m "Description of changes"
```

## Migration Script Structure

Migration scripts are stored in the `versions/` directory. Each script has:

1. An `upgrade()` function that applies the changes
2. A `downgrade()` function that reverts the changes

Example:

```python
def upgrade():
    # Add new table or column
    op.add_column('table_name', sa.Column('new_column', sa.Integer()))

def downgrade():
    # Remove the added column
    op.drop_column('table_name', 'new_column')
```

## Best Practices

1. Always test migrations in a development environment before running in production
2. Create a database backup before running migrations in production
3. Keep migration scripts idempotent when possible
4. Include data migrations in the same script as schema changes when they're related
