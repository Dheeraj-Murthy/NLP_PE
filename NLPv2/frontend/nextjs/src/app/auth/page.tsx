"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      // TODO: Connect to backend auth
      await new Promise((resolve) => setTimeout(resolve, 1000));
      
      // Mock successful auth
      localStorage.setItem("user", JSON.stringify({
        name: isLogin ? "Demo User" : name,
        email,
        role: "user"
      }));
      
      router.push("/dashboard");
    } catch (error) {
      console.error("Auth error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center p-6 bg-[#F7F6F2]">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-xl border border-[#E5E2D9] overflow-hidden">
        <div className="p-8">
          <div className="text-center mb-8">
            <div className="w-12 h-12 bg-[#111827] text-white rounded-xl flex items-center justify-center font-serif font-bold text-2xl mx-auto mb-4">
              §
            </div>
            <h1 className="font-serif text-2xl font-semibold text-[#111827]">
              {isLogin ? "Welcome Back" : "Create Account"}
            </h1>
            <p className="text-[#6B7280] text-sm mt-2">
              {isLogin
                ? "Sign in to access your legal research workspace"
                : "Join LegalRAG to start analyzing documents"}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-[#374151] mb-1">
                  Full Name
                </label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all"
                  placeholder="John Doe"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-[#374151] mb-1">
                Email Address
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all"
                placeholder="counsel@example.com"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-sm font-medium text-[#374151]">
                  Password
                </label>
                {isLogin && (
                  <a href="#" className="text-xs text-[#8B1E28] hover:underline">
                    Forgot password?
                  </a>
                )}
              </div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all"
                placeholder="••••••••"
              />
            </div>

            {isLogin && (
              <div className="flex items-center">
                <input
                  id="remember-me"
                  type="checkbox"
                  className="h-4 w-4 text-[#8B1E28] focus:ring-[#8B1E28] border-[#E5E2D9] rounded cursor-pointer"
                />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-[#6B7280] cursor-pointer">
                  Remember me for 30 days
                </label>
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-[#111827] text-white py-2.5 rounded-lg font-medium hover:bg-[#8B1E28] transition-colors disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                isLogin ? "Sign In" : "Create Account"
              )}
            </button>
          </form>

          <div className="mt-8 text-center">
            <p className="text-sm text-[#6B7280]">
              {isLogin ? "Don't have an account?" : "Already have an account?"}{" "}
              <button
                onClick={() => setIsLogin(!isLogin)}
                className="text-[#8B1E28] font-medium hover:underline focus:outline-none"
              >
                {isLogin ? "Sign up" : "Sign in"}
              </button>
            </p>
          </div>
        </div>
        
        <div className="bg-[#F7F6F2] px-8 py-4 border-t border-[#E5E2D9] text-center">
          <p className="text-xs text-[#6B7280]">
            By continuing, you agree to our{" "}
            <a href="#" className="underline hover:text-[#111827]">Terms of Service</a>
            {" "}and{" "}
            <a href="#" className="underline hover:text-[#111827]">Privacy Policy</a>.
          </p>
        </div>
      </div>
    </div>
  );
}
