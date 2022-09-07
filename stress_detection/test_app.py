from flask import Flask, render_template, url_for
from flask_sqlalchemy import SQLAlchemy as sql
from datetime import datetime


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///test.db'
db = sql(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return "<Task %r>" % self.id

@app.route("/", methods=["POST", "GET"])
def base():
    return render_template("base.html")

if __name__ == "__main__":
    app.run(debug=True)
