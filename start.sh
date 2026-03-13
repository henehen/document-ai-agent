echo #!/bin/bash > start.sh
echo uvicorn server:app --host 0.0.0.0 --port $PORT >> start.sh