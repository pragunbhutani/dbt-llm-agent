"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Dialog,
  DialogBackdrop,
  DialogPanel,
  Menu,
  MenuButton,
  TransitionChild,
} from "@headlessui/react";
import {
  StarIcon,
  HomeIcon,
  XMarkIcon,
  Cog6ToothIcon,
  ChatBubbleLeftRightIcon,
  AcademicCapIcon,
  WrenchScrewdriverIcon,
} from "@heroicons/react/24/outline";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: HomeIcon },
  {
    name: "Knowledge Base",
    href: "/dashboard/knowledge-base",
    icon: AcademicCapIcon,
  },
  {
    name: "Chats",
    href: "/dashboard/chats",
    icon: ChatBubbleLeftRightIcon,
  },
  {
    name: "Integrations",
    href: "/dashboard/integrations",
    icon: WrenchScrewdriverIcon,
  },
  {
    name: "Settings",
    href: "/dashboard/settings",
    icon: Cog6ToothIcon,
  },
];

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(" ");
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const pathname = usePathname();

  return (
    <>
      <div>
        <Dialog
          open={sidebarOpen}
          onClose={setSidebarOpen}
          className="relative z-50 lg:hidden"
        >
          <DialogBackdrop
            transition
            className="fixed inset-0 bg-gray-900/80 transition-opacity duration-300 ease-linear data-closed:opacity-0"
          />

          <div className="fixed inset-0 flex">
            <DialogPanel
              transition
              className="relative mr-16 flex w-full max-w-xs flex-1 transform transition duration-300 ease-in-out data-closed:-translate-x-full"
            >
              <TransitionChild>
                <div className="absolute top-0 left-full flex w-16 justify-center pt-5 duration-300 ease-in-out data-closed:opacity-0">
                  <button
                    type="button"
                    onClick={() => setSidebarOpen(false)}
                    className="-m-2.5 p-2.5"
                  >
                    <span className="sr-only">Close sidebar</span>
                    <XMarkIcon
                      aria-hidden="true"
                      className="size-6 text-white"
                    />
                  </button>
                </div>
              </TransitionChild>

              {/* Sidebar component, swap this element with another sidebar if you like */}
              <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white px-6 pb-4">
                <div className="flex h-16 shrink-0 items-center">
                  <StarIcon className="h-8 w-8 text-indigo-600" />
                </div>
                <nav className="flex flex-1 flex-col">
                  <ul role="list" className="flex flex-1 flex-col gap-y-7">
                    <li>
                      <ul role="list" className="-mx-2 space-y-1">
                        {navigation.map((item) => (
                          <li key={item.name}>
                            <Link
                              href={item.href}
                              className={classNames(
                                pathname.startsWith(item.href)
                                  ? "bg-gray-50 text-indigo-600"
                                  : "text-gray-700 hover:bg-gray-50 hover:text-indigo-600",
                                "group flex gap-x-3 rounded-md p-2 text-sm/6 font-semibold"
                              )}
                            >
                              <item.icon
                                aria-hidden="true"
                                className={classNames(
                                  pathname.startsWith(item.href)
                                    ? "text-indigo-600"
                                    : "text-gray-400 group-hover:text-indigo-600",
                                  "size-6 shrink-0"
                                )}
                              />
                              {item.name}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    </li>
                  </ul>
                </nav>
              </div>
            </DialogPanel>
          </div>
        </Dialog>

        {/* Static sidebar for desktop */}
        <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-24 lg:flex-col">
          {/* Sidebar component, swap this element with another sidebar if you like */}
          <div className="flex grow flex-col overflow-y-auto border-r border-gray-200 bg-white">
            <div className="flex h-16 shrink-0 items-center justify-center bg-indigo-600">
              <StarIcon className="h-8 w-8 text-white" />
            </div>
            <nav className="flex flex-1 flex-col p-2">
              <ul role="list" className="flex flex-1 flex-col gap-y-7">
                <li>
                  <ul role="list" className="space-y-1">
                    {navigation.map((item) => (
                      <li key={item.name}>
                        <Link
                          href={item.href}
                          className={classNames(
                            pathname.startsWith(item.href)
                              ? "bg-gray-50 text-indigo-600"
                              : "text-gray-700 hover:bg-gray-50 hover:text-indigo-600",
                            "group flex w-full flex-col items-center justify-center rounded-md py-2 min-h-20 text-xs font-semibold"
                          )}
                        >
                          <item.icon
                            aria-hidden="true"
                            className={classNames(
                              pathname.startsWith(item.href)
                                ? "text-indigo-600"
                                : "text-gray-400 group-hover:text-indigo-600",
                              "size-7 shrink-0"
                            )}
                          />
                          <span className="mt-2 text-center font-light text-xs leading-none">
                            {item.name}
                          </span>
                        </Link>
                      </li>
                    ))}
                  </ul>
                </li>
                <li className="mt-auto">
                  <Menu as="div" className="relative">
                    <MenuButton className="flex w-full items-center justify-center rounded-md py-3 text-sm/6 font-semibold text-gray-900 hover:bg-gray-50">
                      <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-sm font-medium text-white">
                        PB
                      </span>
                    </MenuButton>
                  </Menu>
                </li>
              </ul>
            </nav>
          </div>
        </div>

        <div className="lg:pl-24">
          <main>
            <div className="px-0.5 sm:px-1 lg:px-2">{children}</div>
          </main>
        </div>
      </div>
    </>
  );
}
