from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Property(db.Model):
    __tablename__ = 'properties'
    PropertyID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PropType = db.Column(db.String(), nullable=True)
    nbhd = db.Column(db.Integer, nullable=True)
    Style = db.Column(db.String(), nullable=True)
    Stories = db.Column(db.Integer, nullable=True)
    Year_Built = db.Column(db.Integer, nullable=True)
    Rooms = db.Column(db.Integer(), nullable=True)
    FinishedSqft = db.Column(db.Integer, nullable=True)
    Units = db.Column(db.Integer, nullable=True)
    Bdrms = db.Column(db.Integer, nullable=True)
    Fbath = db.Column(db.Integer, nullable=True)
    Hbath = db.Column(db.Integer, nullable=True)
    Sale_price = db.Column(db.Integer, nullable=True)
    RICH = db.Column(db.String())
    index = db.Column(db.Integer, nullable=True)
