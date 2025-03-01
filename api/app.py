from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

# Vercel expects a `handler` function as the entry point
def handler(request):
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.serving import run_simple

    # We need to wrap our Flask app in a WSGI server
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app)
    return app(request)
