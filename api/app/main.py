import json
from http import HTTPStatus, HTTPMethod
from wsgiref.simple_server import make_server
import traceback

from mip_model import run_mip_model


def app(environ, start_response):
    method = environ["REQUEST_METHOD"]
    path = environ["PATH_INFO"]
    status = HTTPStatus.OK
    headers = [("Content-type", "application/json")]
    response = {}

    if method == HTTPMethod.GET and path == "/":
        response = {"message": "Hello, World!"}
    elif method == HTTPMethod.POST and path == "/mip":
        try:
            ok, result = post_mip(environ)
            if ok:
                response = result
            else:
                status = HTTPStatus.BAD_REQUEST
                response = result
        except Exception as err:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            response = {"message": str(err)}
            print(err)
            traceback.print_exc()
    else:
        status = HTTPStatus.NOT_FOUND
        response = {"message": "Not Found"}

    start_response(f"{status.value} {status.phrase}", headers)
    return [json.dumps(response).encode("utf-8")]


def post_mip(environ):
    # Get the content length
    content_length = int(environ.get("CONTENT_LENGTH", 0))
    # Read the body from the input stream
    input_str = environ["wsgi.input"].read(content_length)
    return run_mip_model(input_str)


def main():
    host = "0.0.0.0"
    port = 8000
    with make_server(host, port, app) as httpd:
        print(f"Serving on {host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
