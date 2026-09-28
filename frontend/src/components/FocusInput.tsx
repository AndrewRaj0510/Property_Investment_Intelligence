"use client";

import { CSSProperties, useState } from "react";

export function FocusInput({
  value,
  onChange,
  onKeyDown,
  placeholder,
  type = "text",
  style,
  focusStyle,
  disabled = false,
  maxLength,
}: {
  value: string;
  onChange: (v: string) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  placeholder?: string;
  type?: string;
  style: CSSProperties;
  focusStyle?: CSSProperties;
  disabled?: boolean;
  maxLength?: number;
}) {
  const [focus, setFocus] = useState(false);
  return (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
      disabled={disabled}
      maxLength={maxLength}
      onFocus={() => setFocus(true)}
      onBlur={() => setFocus(false)}
      style={{ ...style, ...(focus ? focusStyle : {}) }}
    />
  );
}
