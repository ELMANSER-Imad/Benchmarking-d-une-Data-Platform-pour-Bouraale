
# /bin/bash
docker container stop $(docker ps -aq)
docker container rm $(docker ps -aq)
docker image rm $(docker images -q)
docker volume rm $(docker volume ls -aq)
docker system prune -f
