# Database

SQLite is the default local database and is created automatically as `federated_healthcare.db`.
SQLAlchemy models cover hospitals, experiments, experiment metrics, training rounds, model
versions, predictions, SHAP explanations, privacy settings and audit logs.

The database is seeded from the verified project artifacts; no synthetic patient records are
created by the seed process.
