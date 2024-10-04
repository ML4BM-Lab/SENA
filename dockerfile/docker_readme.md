
## json file
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Remote Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/path/inside/container"
        }
      ],
      "justMyCode": true
    }
  ]
}

## install 
RUN pip install debugpy

## how to debug
docker run -p 5678:5678 -v /path/to/local:/path/inside/container -it your_image

## run script 
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client your_script.py