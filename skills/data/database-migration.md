---
name: database-migration
description: Execute database migrations across ORMs with zero-downtime strategies, data transformation, and rollback procedures. Use when migrating databases or changing schemas.
trigger_keywords: [database migration, schema migration, sequelize, typeorm, prisma migrate, zero downtime, rollback, db:migrate]
---

# Database Migration

Master database schema and data migrations across ORMs (Sequelize, TypeORM, Prisma).

## When to Use

- Migrating between different ORMs
- Performing schema transformations
- Moving data between databases
- Zero-downtime deployments
- Database version upgrades

## Core Concepts

- **Idempotency**: Migrations must be rerunnable without side effects - use IF NOT EXISTS, check for column existence before adding, and handle already-applied states gracefully
- **Backward Compatibility**: Never break running applications - add new columns with defaults, deprecate before removing, and support reading both old and new schemas during transitions
- **Transaction Safety**: Wrap DDL operations in transactions where supported (PostgreSQL), split large data migrations into batched transactions, and always test rollback procedures
- **Schema Versioning**: Track migration history with timestamps and checksums, maintain linear migration order, and never modify already-deployed migrations - create new ones instead
- **Data Preservation**: Always backup before migration, use COPY/INSERT with explicit column lists, validate row counts and data integrity post-migration

## ORM Migrations

### Sequelize
```javascript
module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('users', {
      id: { type: Sequelize.INTEGER, primaryKey: true, autoIncrement: true },
      email: { type: Sequelize.STRING, unique: true, allowNull: false },
      createdAt: Sequelize.DATE
    });
  },
  down: async (queryInterface) => {
    await queryInterface.dropTable('users');
  }
};
// Run: npx sequelize-cli db:migrate
```

### TypeORM
```typescript
export class CreateUsers1701234567 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.createTable(new Table({
      name: 'users',
      columns: [
        { name: 'id', type: 'int', isPrimary: true, isGenerated: true },
        { name: 'email', type: 'varchar', isUnique: true }
      ]
    }));
  }
  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.dropTable('users');
  }
}
// Run: npm run typeorm migration:run
```

### Prisma
```prisma
model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  createdAt DateTime @default(now())
}
// npx prisma migrate dev --name create_users
```

## Zero-Downtime Rename Column

```javascript
// Step 1: Add new column
await queryInterface.addColumn('users', 'full_name', { type: Sequelize.STRING });
await queryInterface.sequelize.query('UPDATE users SET full_name = name');

// Step 2: Update application to use new column

// Step 3: Remove old column
await queryInterface.removeColumn('users', 'name');
```

## Transaction-Based Migrations

```javascript
module.exports = {
  up: async (queryInterface, Sequelize) => {
    const transaction = await queryInterface.sequelize.transaction();
    try {
      await queryInterface.addColumn('users', 'verified',
        { type: Sequelize.BOOLEAN, defaultValue: false }, { transaction });
      await queryInterface.sequelize.query(
        'UPDATE users SET verified = true WHERE email_verified_at IS NOT NULL',
        { transaction });
      await transaction.commit();
    } catch (error) {
      await transaction.rollback();
      throw error;
    }
  }
};
```

## Blue-Green Deployment

1. **Phase 1**: Add new column (backward compatible)
2. **Phase 2**: Deploy code writing to both columns
3. **Phase 3**: Backfill data
4. **Phase 4**: Deploy code reading new column
5. **Phase 5**: Remove old column

## Cross-Database Migrations

```javascript
const dialectName = queryInterface.sequelize.getDialect();
if (dialectName === 'mysql') {
  // MySQL JSON type
} else if (dialectName === 'postgres') {
  // PostgreSQL JSONB type
}
```

## Best Practices

1. **Always Provide Rollback**: Every up() needs down()
2. **Test Migrations**: Test on staging first
3. **Use Transactions**: Atomic migrations when possible
4. **Backup First**: Always backup before migration
5. **Small Changes**: Break into incremental steps
6. **Idempotent**: Migrations should be rerunnable

## Common Pitfalls

- Not testing rollback procedures
- Making breaking changes without downtime strategy
- Forgetting to handle NULL values
- Not considering index performance
- Ignoring foreign key constraints
- Migrating too much data at once
