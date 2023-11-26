import React, { useRef, useEffect, useCallback, useState } from "react";
import { Paper, Typography } from "@mui/material";

function getRandomColor() {
  const letters = "0123456789ABCDEF";
  let color = "#";
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [currentColor, setCurrentColor] = useState("#000000");
  const [currentPrompt, setCurrentPrompt] = useState(
    "psychedelic structure, high quality"
  );

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
        image.src = "images/white.jpg";
      }
    }
  }, []);

  useEffect(() => {
    const colorChangeInterval = setInterval(() => {
      setCurrentColor(getRandomColor());
    }, 10000);

    return () => clearInterval(colorChangeInterval);
  }, []);

  const sendUpdatePrompt = useCallback(async () => {
    try {
      const response = await fetch("http://127.0.0.1:9090/update_prompt", {
        method: "GET",
      });
      const data = await response.json();
      setCurrentPrompt(data.prompt);
    } catch (error) {
      console.error("Error sending image:", error);
    }
  }, []);

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      sendUpdatePrompt();
    }, 45000);
  }, [sendUpdatePrompt]);

  const sendImage = useCallback(async (base64Image: string) => {
    try {
      const response = await fetch("http://127.0.0.1:9090/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base64_image: base64Image }),
      });
      const data = await response.json();
      updateCanvas(data.base64_image);
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
    }, 750);

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
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      };
      image.src = `data:image/jpeg;base64,${newImageBase64}`;
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
      // line
      ctx.strokeStyle = currentColor;
      ctx.lineWidth = 150;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      // shadow
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 10;
      ctx.shadowOffsetX = 5;
      ctx.shadowOffsetY = 5;

      // draw
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
    <div
      className="App"
      style={{
        backgroundColor: "#282c34",
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        margin: "0",
        color: "#ffffff",
      }}
    >
      <div
        style={{
          backgroundColor: "#282c34",
          alignItems: "center",
          justifyContent: "center",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography
          variant="h6"
          style={{ color: "#ffffff", padding: "4vh", fontFamily: "Kanit" }}
        >
          {currentPrompt}
        </Typography>
        <canvas
          ref={canvasRef}
          width="640px"
          height="640px"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
          style={{
            border: "1px solid #000000",
            borderRadius: "10px",
            height: "80vh",
          }}
        />
      </div>
      <img ref={imageRef} style={{ display: "none" }} alt="background" />
    </div>
  );
}
export default App;
