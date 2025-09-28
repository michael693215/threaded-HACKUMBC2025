from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/folder")
def folder_page():
    return render_template("folder.html")

@app.route("/outfits/create")
def outfits_create():
    return render_template("create.html")

@app.route("/outfits/saved")
def outfits_saved():
    return render_template("saved.html")

if __name__ == "__main__":
    app.run(debug=True)
