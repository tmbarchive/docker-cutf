Dockerfile.built: Dockerfile
	docker build . -t tf
	touch Dockerfile.built
