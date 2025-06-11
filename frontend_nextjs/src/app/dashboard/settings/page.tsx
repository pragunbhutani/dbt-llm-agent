import Heading from "@/components/heading";

export default function SettingsPage() {
  return (
    <>
      <div className="flex h-16 items-center border-b border-gray-200">
        <Heading
          title="Settings"
          subtitle="Manage your account and workspace settings."
        />
      </div>
      <div className="p-4">
        <p>Settings content will go here.</p>
      </div>
    </>
  );
}
