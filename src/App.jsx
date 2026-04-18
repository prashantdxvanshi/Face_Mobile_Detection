"use client";
import { useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import * as blazeface from "@tensorflow-models/blazeface";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

export default function App() {
  const videoRef = useRef(null);

  useEffect(() => {
    startCamera();
    runDetection();
  }, []);

  // 🎥 Start webcam
  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
  };

  // 🧠 Load models and detect
  const runDetection = async () => {
    await tf.setBackend("webgl");

    const faceModel = await blazeface.load();
    const objectModel = await cocoSsd.load();

    console.log("Models Loaded");

    setInterval(async () => {
      if (!videoRef.current) return;

      const video = videoRef.current;

      // 👤 FACE DETECTION
      const faces = await faceModel.estimateFaces(video, false);

      if (faces.length === 0) {
        alert("❌ No face detected");
      } else if (faces.length === 1) {
        console.log("✅ One face detected");
      } else {
        alert("🚨 Multiple faces detected");
      }

      // 📱 OBJECT DETECTION
      const objects = await objectModel.detect(video);

      objects.forEach((obj) => {
        if (obj.score > 0.6) {
          console.log(`Detected: ${obj.class} (${obj.score.toFixed(2)})`);

          if (obj.class === "cell phone") {
            alert("🚨 Mobile detected!");
          }
        }
      });

    }, 500); // runs every 0.5 sec
  };

  return (
    <div>
      <h2>AI Detection Running...</h2>
      <video ref={videoRef} autoPlay playsInline width="400" />
    </div>
  );
}