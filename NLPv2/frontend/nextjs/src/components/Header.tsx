"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Header() {
    const pathname = usePathname();

    const navLinks = [
        { name: "Chat", path: "/" },
        { name: "Dashboard", path: "/dashboard" },
        { name: "Profile", path: "/profile" },
    ];

    return (
        <header className="bg-white border-b border-[#E5E2D9] px-6 py-3 flex items-center justify-between shadow-sm sticky top-0 z-50">

            <Link href="/" className="font-serif text-xl font-semibold text-[#111827] hover:text-[#8B1E28] transition-colors">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-[#111827] text-white rounded flex items-center justify-center font-serif font-bold text-lg">
                        §
                    </div>
                    LegalRAG
                    <span className="text-xs uppercase tracking-wider font-semibold text-[#8B1E28] bg-[#8B1E28]/10 px-2 py-0.5 rounded-full hidden sm:inline-block">
                        AI Counsel
                    </span>
                </div>
            </Link>

            <nav className="flex items-center gap-6">
                <div className="hidden md:flex items-center gap-6">
                    {navLinks.map((link) => (
                        <Link
                            key={link.path}
                            href={link.path}
                            className={`text-sm font-medium transition-colors ${
pathname === link.path
? "text-[#8B1E28] border-b-2 border-[#8B1E28] py-1"
: "text-[#6B7280] hover:text-[#111827] py-1"
}`}
                        >
                            {link.name}
                        </Link>
                    ))}
                </div>
                <div className="flex items-center gap-3 border-l border-[#E5E2D9] pl-6">
                    <Link
                        href="/auth"
                        className="text-sm font-medium text-[#111827] hover:text-[#8B1E28] transition-colors"
                    >
                        Sign In
                    </Link>
                </div>
            </nav>
        </header>
    );
}
