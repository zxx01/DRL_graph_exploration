IMAGE_NAME := drl2
# CATKIN_WS_DIR_ := "$(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)"

all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build all images"
    # @echo "   1. make pull             - pull all images"
	@echo "   1. make clean              - remove all images"

build:
	@sudo docker build --no-cache --network=host --tag $(IMAGE_NAME) -f ./DRL2.Dockerfile .

# pull:
# 	@sudo docker pull $(IMAGE_NAME)

clean:
	@sudo docker rmi -f $(IMAGE_NAME)
