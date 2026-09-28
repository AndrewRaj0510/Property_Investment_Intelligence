"use client";

import { CSSProperties, useState } from "react";

export function HoverButton({
  style,
  hoverStyle,
  activeStyle,
  onClick,
  children,
  type = "button",
  title,
  disabled = false,
}: {
  style: CSSProperties;
  hoverStyle?: CSSProperties;
  activeStyle?: CSSProperties;
  onClick?: () => void;
  children: React.ReactNode;
  type?: "button" | "submit";
  title?: string;
  disabled?: boolean;
}) {
  const [hover, setHover] = useState(false);
  const [active, setActive] = useState(false);
  const interactive = !disabled;
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => {
        setHover(false);
        setActive(false);
      }}
      onMouseDown={() => setActive(true)}
      onMouseUp={() => setActive(false)}
      style={{
        ...style,
        ...(interactive && hover ? hoverStyle : {}),
        ...(interactive && active ? activeStyle : {}),
      }}
    >
      {children}
    </button>
  );
}
