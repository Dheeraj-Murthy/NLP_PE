"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface User {
  name: string;
  email: string;
  role: string;
}

export default function DashboardPage() {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    } else {
      setUser({ name: "Guest User", email: "guest@example.com", role: "guest" });
    }
  }, []);

  const stats = [
    { label: "Total Queries", value: "1,248", trend: "+12%", icon: "🔍" },
    { label: "Documents Analyzed", value: "342", trend: "+5%", icon: "📄" },
    { label: "Hours Saved", value: "86", trend: "+18%", icon: "⏱️" },
    { label: "Active Cases", value: "14", trend: "0%", icon: "⚖️" },
  ];

  const recentChats = [
    { id: 1, title: "Karnataka Education Act Analysis", date: "Today, 10:42 AM", status: "Completed" },
    { id: 2, title: "Vijay Kumar v. State Precedents", date: "Yesterday, 3:15 PM", status: "Completed" },
    { id: 3, title: "Contract Review: TechCorp NDA", date: "Mar 14, 2026", status: "In Progress" },
    { id: 4, title: "Property Dispute Case Law", date: "Mar 12, 2026", status: "Completed" },
    { id: 5, title: "Corporate Governance Guidelines", date: "Mar 10, 2026", status: "Completed" },
  ];

  const usageData = [
    { day: "Mon", value: 45 },
    { day: "Tue", value: 62 },
    { day: "Wed", value: 38 },
    { day: "Thu", value: 85 },
    { day: "Fri", value: 54 },
    { day: "Sat", value: 12 },
    { day: "Sun", value: 8 },
  ];

  const maxUsage = Math.max(...usageData.map(d => d.value));

  return (
    <div className="flex-1 overflow-y-auto bg-[#F7F6F2] p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="font-serif text-3xl font-semibold text-[#111827]">
              Welcome back, {user?.name.split(' ')[0] || 'Counsel'}
            </h1>
            <p className="text-[#6B7280] mt-1">
              Here's an overview of your legal research activities.
            </p>
          </div>
          <Link 
            href="/"
            className="bg-[#111827] text-white px-5 py-2.5 rounded-lg font-medium hover:bg-[#8B1E28] transition-colors inline-flex items-center gap-2 shadow-sm"
          >
            <span>New Research Query</span>
            <span>→</span>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <div key={index} className="bg-white p-6 rounded-xl border border-[#E5E2D9] shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 bg-[#F7F6F2] rounded-lg flex items-center justify-center text-xl">
                  {stat.icon}
                </div>
                <span className={`text-sm font-medium ${stat.trend.startsWith('+') ? 'text-green-600' : 'text-[#6B7280]'}`}>
                  {stat.trend}
                </span>
              </div>
              <h3 className="text-[#6B7280] text-sm font-medium">{stat.label}</h3>
              <p className="text-3xl font-serif font-semibold text-[#111827] mt-1">{stat.value}</p>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 bg-white rounded-xl border border-[#E5E2D9] shadow-sm overflow-hidden flex flex-col">
            <div className="p-6 border-b border-[#E5E2D9] flex items-center justify-between">
              <h2 className="font-serif text-xl font-semibold text-[#111827]">Recent Research Sessions</h2>
              <button className="text-sm text-[#8B1E28] font-medium hover:underline">View All</button>
            </div>
            <div className="flex-1 overflow-y-auto">
              <ul className="divide-y divide-[#E5E2D9]">
                {recentChats.map((chat) => (
                  <li key={chat.id} className="p-6 hover:bg-[#F7F6F2] transition-colors group cursor-pointer">
                    <div className="flex items-center justify-between">
                      <div className="flex items-start gap-4">
                        <div className="w-8 h-8 rounded-full bg-[#111827] text-white flex items-center justify-center font-serif text-sm flex-shrink-0 mt-0.5">
                          §
                        </div>
                        <div>
                          <h3 className="font-medium text-[#111827] group-hover:text-[#8B1E28] transition-colors">
                            {chat.title}
                          </h3>
                          <p className="text-sm text-[#6B7280] mt-1">{chat.date}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                          chat.status === 'Completed' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-amber-100 text-amber-800'
                        }`}>
                          {chat.status}
                        </span>
                        <span className="text-[#C5A880] opacity-0 group-hover:opacity-100 transition-opacity">
                          →
                        </span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="space-y-8">
            <div className="bg-white rounded-xl border border-[#E5E2D9] shadow-sm p-6">
              <h2 className="font-serif text-xl font-semibold text-[#111827] mb-6">Weekly Usage</h2>
              <div className="flex items-end justify-between h-48 gap-2">
                {usageData.map((data, index) => (
                  <div key={index} className="flex flex-col items-center gap-2 flex-1">
                    <div className="w-full bg-[#F7F6F2] rounded-t-md relative group h-full flex items-end">
                      <div 
                        className="w-full bg-[#111827] rounded-t-md transition-all duration-500 group-hover:bg-[#8B1E28]"
                        style={{ height: `${(data.value / maxUsage) * 100}%` }}
                      ></div>
                      <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-[#111827] text-white text-xs py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                        {data.value} queries
                      </div>
                    </div>
                    <span className="text-xs font-medium text-[#6B7280]">{data.day}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl border border-[#E5E2D9] shadow-sm p-6">
              <h2 className="font-serif text-xl font-semibold text-[#111827] mb-4">Profile Summary</h2>
              <div className="flex items-center gap-4 mb-6">
                <div className="w-16 h-16 bg-[#C5A880] text-white rounded-full flex items-center justify-center text-2xl font-serif font-bold">
                  {user?.name.charAt(0) || 'U'}
                </div>
                <div>
                  <h3 className="font-medium text-[#111827] text-lg">{user?.name || 'User'}</h3>
                  <p className="text-sm text-[#6B7280]">{user?.email || 'user@example.com'}</p>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-[#6B7280]">Plan</span>
                  <span className="font-medium text-[#111827]">Enterprise Legal</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#6B7280]">Member Since</span>
                  <span className="font-medium text-[#111827]">Jan 2026</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#6B7280]">Storage Used</span>
                  <span className="font-medium text-[#111827]">2.4 GB / 10 GB</span>
                </div>
              </div>
              <Link 
                href="/profile"
                className="mt-6 w-full block text-center py-2 border border-[#E5E2D9] rounded-lg text-sm font-medium text-[#374151] hover:bg-[#F7F6F2] transition-colors"
              >
                Manage Profile
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
