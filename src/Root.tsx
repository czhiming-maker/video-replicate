import React from "react";
import { Composition } from "remotion";
import { AutumnScene } from "./AutumnScene";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Autumn"
      component={AutumnScene}
      durationInFrames={122}
      fps={24}
      width={720}
      height={1280}
    />
  );
};
