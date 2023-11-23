.PHONY: install-rye build up

build:
	sudo cp -r src docker \
	&& sudo cp -r view docker \
	&& sudo cp -r server docker \
	&& sudo cp requirements.lock docker \
	&& sudo cp pyproject.toml docker \
	&& sudo cp README.md docker \
	&& cd docker \
	&& docker-compose build;

up:
	if [ -e "docker/pyproject.toml" ]; then \
		sudo rm -rf docker/src; \
		sudo rm -rf docker/view; \
		sudo rm -rf docker/server; \
		sudo rm -rf docker/requirements.lock; \
		sudo rm -rf docker/pyproject.toml; \
		sudo rm -rf docker/README.md; \
	fi \
	&& cd docker && docker-compose up;

install-rye:
	curl -sSf https://rye-up.com/get | bash \
	&& echo 'source "$$HOME/.rye/env"' >> ~/.bashrc
