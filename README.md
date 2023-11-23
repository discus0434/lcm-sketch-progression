# LCM Sketch Progression

**LCM Sketch Progression is a web app that performs real-time image generation and allows you to actively intervene in the progress with a brush.**

<p align="center">
  <img src="./assets/sample.gif" width=85%>
</p>

## Features

- The app generates an image based on the previous state of the image.
- You can intervene in the progress of the image generation by drawing on the canvas with a brush.
- The prompt is updated periodically and randomly, so the generated image keeps changing uniquely.


## Installation

### 1. Build the Docker images

```bash
make build
```

### 2. Run the app

```bash
make up
```

### 3. Access the app

Open `http://<PUBLIC_IP or 0.0.0.0>:10356` in your browser.

## Configuration

You can configure the app's behavior at `server/config.py`.
For example, if you have less than 8GB of VRAM, you can use xformers with set as below in `server/config.py`.

```diff
-torch_compile: bool = True
-xformers: bool = False
+torch_compile: bool = False
+xformers: bool = True
```

## Requirements

- NVIDIA GPU more than 6GB of VRAM
- Docker

## License

This project is licensed under the [Affero General Public License v3.0](./LICENSE).
