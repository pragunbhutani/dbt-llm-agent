import React from "react";

interface HeadingProps {
  title: string;
  subtitle?: string;
}

const Heading: React.FC<HeadingProps> = ({ title, subtitle }) => {
  return (
    <div className="px-2">
      <h1 className="text-xl font-semibold leading-6 text-gray-900">{title}</h1>
      {subtitle && (
        <p className="mt-1 max-w-4xl text-sm text-gray-500">{subtitle}</p>
      )}
    </div>
  );
};

export default Heading;
