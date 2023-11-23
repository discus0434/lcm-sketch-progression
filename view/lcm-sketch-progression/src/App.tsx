import React, { useRef, useEffect, useCallback } from "react";

function DrawingApp() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      const image = imageRef.current;
      if (image) {
        image.onload = () => {
          canvas.width = image.width;
          canvas.height = image.height;
          ctx.drawImage(image, 0, 0, image.width, image.height);
        };
        image.src = "images/myImage.jpg";
      }
    }
  }, []);

  const sendImage = useCallback(async (base64Image: string) => {
    try {
      const response = await fetch("http://127.0.0.1:9090/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image }),
      });
      const data = await response.json();
      updateCanvas(data.image);
    } catch (error) {
      console.error("Error sending image:", error);
    }
  }, []);

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      if (canvasRef.current) {
        const base64Image = canvasRef.current.toDataURL("image/png");
        sendImage(base64Image);
      }
    }, 1000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [sendImage]);

  const updateCanvas = (newImageBase64: string) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      const image = new Image();
      image.onload = () => {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      };
      image.src = newImageBase64;
    }
  };

  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDrawing = true;
    const { offsetX, offsetY } = e.nativeEvent;
    lastX = offsetX;
    lastY = offsetY;
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
      ctx.stroke();
      lastX = e.nativeEvent.offsetX;
      lastY = e.nativeEvent.offsetY;
    }
  };

  const stopDrawing = () => {
    isDrawing = false;
  };

  return (
    <div>
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseOut={stopDrawing}
      />
      <img ref={imageRef} style={{ display: "none" }} alt="background" />
    </div>
  );
}

export default DrawingApp;
