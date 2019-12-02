from flask import Flask, escape, request, make_response

app = Flask(__name__)


@app.route("/download/<path>")
def download_model(path=None):
    try:
        with open(path, 'rb') as f:
            response = make_response(f.read())
        response.headers["Content-Disposition"] = "attachment; filename=%s" % path
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(e)
