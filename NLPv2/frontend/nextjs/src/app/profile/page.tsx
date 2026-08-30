"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface User {
  name: string;
  email: string;
  role: string;
}

export default function ProfilePage() {
  const [user, setUser] = useState<User | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [notifications, setNotifications] = useState({
    emailAlerts: true,
    caseUpdates: true,
    marketing: false,
  });
  const [theme, setTheme] = useState("light");
  const router = useRouter();

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      const parsedUser = JSON.parse(storedUser);
      setUser(parsedUser);
      setName(parsedUser.name);
      setEmail(parsedUser.email);
    } else {
      // Mock user if not logged in
      const mockUser = { name: "Demo User", email: "demo@example.com", role: "user" };
      setUser(mockUser);
      setName(mockUser.name);
      setEmail(mockUser.email);
    }
  }, []);

  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    const updatedUser = { ...user, name, email } as User;
    setUser(updatedUser);
    localStorage.setItem("user", JSON.stringify(updatedUser));
    setIsEditing(false);
    // Show success toast here in a real app
  };

  const handlePasswordChange = (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      alert("Passwords do not match");
      return;
    }
    // Handle password change logic here
    setCurrentPassword("");
    setNewPassword("");
    setConfirmPassword("");
    alert("Password updated successfully");
  };

  const handleLogout = () => {
    localStorage.removeItem("user");
    router.push("/auth");
  };

  if (!user) return null;

  return (
    <div className="flex-1 overflow-y-auto bg-[#F7F6F2] p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        <div>
          <h1 className="font-serif text-3xl font-semibold text-[#111827]">
            Account Settings
          </h1>
          <p className="text-[#6B7280] mt-1">
            Manage your profile, preferences, and security settings.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Sidebar Navigation */}
          <div className="md:col-span-1 space-y-2">
            <button className="w-full text-left px-4 py-3 rounded-lg bg-white border border-[#E5E2D9] font-medium text-[#8B1E28] shadow-sm flex items-center gap-3">
              <span className="text-lg">👤</span> Profile Information
            </button>
            <button className="w-full text-left px-4 py-3 rounded-lg hover:bg-white border border-transparent hover:border-[#E5E2D9] font-medium text-[#6B7280] hover:text-[#111827] transition-all flex items-center gap-3">
              <span className="text-lg">🔒</span> Security
            </button>
            <button className="w-full text-left px-4 py-3 rounded-lg hover:bg-white border border-transparent hover:border-[#E5E2D9] font-medium text-[#6B7280] hover:text-[#111827] transition-all flex items-center gap-3">
              <span className="text-lg">🔔</span> Notifications
            </button>
            <button className="w-full text-left px-4 py-3 rounded-lg hover:bg-white border border-transparent hover:border-[#E5E2D9] font-medium text-[#6B7280] hover:text-[#111827] transition-all flex items-center gap-3">
              <span className="text-lg">⚙️</span> Preferences
            </button>
          </div>

          {/* Main Content Area */}
          <div className="md:col-span-2 space-y-8">
            
            {/* Profile Section */}
            <div className="bg-white rounded-xl border border-[#E5E2D9] shadow-sm overflow-hidden">
              <div className="p-6 border-b border-[#E5E2D9] flex items-center justify-between">
                <h2 className="font-serif text-xl font-semibold text-[#111827]">Profile Information</h2>
                {!isEditing && (
                  <button 
                    onClick={() => setIsEditing(true)}
                    className="text-sm text-[#8B1E28] font-medium hover:underline"
                  >
                    Edit Profile
                  </button>
                )}
              </div>
              
              <div className="p-6">
                <div className="flex items-center gap-6 mb-8">
                  <div className="w-20 h-20 bg-[#C5A880] text-white rounded-full flex items-center justify-center text-3xl font-serif font-bold shadow-inner">
                    {user.name.charAt(0)}
                  </div>
                  <div>
                    <button className="px-4 py-2 border border-[#E5E2D9] rounded-lg text-sm font-medium text-[#374151] hover:bg-[#F7F6F2] transition-colors">
                      Change Avatar
                    </button>
                    <p className="text-xs text-[#6B7280] mt-2">JPG, GIF or PNG. Max size of 2MB.</p>
                  </div>
                </div>

                <form onSubmit={handleSaveProfile} className="space-y-5">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div>
                      <label className="block text-sm font-medium text-[#374151] mb-1">
                        Full Name
                      </label>
                      <input
                        type="text"
                        disabled={!isEditing}
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all disabled:opacity-70 disabled:cursor-not-allowed"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-[#374151] mb-1">
                        Email Address
                      </label>
                      <input
                        type="email"
                        disabled={!isEditing}
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all disabled:opacity-70 disabled:cursor-not-allowed"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-[#374151] mb-1">
                      Role / Title
                    </label>
                    <input
                      type="text"
                      disabled
                      value="Senior Counsel"
                      className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg opacity-70 cursor-not-allowed"
                    />
                    <p className="text-xs text-[#6B7280] mt-1">Contact administrator to change your role.</p>
                  </div>

                  {isEditing && (
                    <div className="flex justify-end gap-3 pt-4 border-t border-[#E5E2D9]">
                      <button
                        type="button"
                        onClick={() => {
                          setIsEditing(false);
                          setName(user.name);
                          setEmail(user.email);
                        }}
                        className="px-5 py-2.5 border border-[#E5E2D9] rounded-lg text-sm font-medium text-[#374151] hover:bg-[#F7F6F2] transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="px-5 py-2.5 bg-[#111827] text-white rounded-lg text-sm font-medium hover:bg-[#8B1E28] transition-colors"
                      >
                        Save Changes
                      </button>
                    </div>
                  )}
                </form>
              </div>
            </div>

            {/* Security Section */}
            <div className="bg-white rounded-xl border border-[#E5E2D9] shadow-sm overflow-hidden">
              <div className="p-6 border-b border-[#E5E2D9]">
                <h2 className="font-serif text-xl font-semibold text-[#111827]">Security</h2>
                <p className="text-sm text-[#6B7280] mt-1">Update your password and secure your account.</p>
              </div>
              
              <div className="p-6">
                <form onSubmit={handlePasswordChange} className="space-y-5">
                  <div>
                    <label className="block text-sm font-medium text-[#374151] mb-1">
                      Current Password
                    </label>
                    <input
                      type="password"
                      required
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                      className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all"
                    />
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div>
                      <label className="block text-sm font-medium text-[#374151] mb-1">
                        New Password
                      </label>
                      <input
                        type="password"
                        required
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-[#374151] mb-1">
                        Confirm New Password
                      </label>
                      <input
                        type="password"
                        required
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        className="w-full px-4 py-2.5 bg-[#F7F6F2] border border-[#E5E2D9] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B1E28]/20 focus:border-[#8B1E28] transition-all"
                      />
                    </div>
                  </div>

                  <div className="flex justify-end pt-4">
                    <button
                      type="submit"
                      disabled={!currentPassword || !newPassword || !confirmPassword}
                      className="px-5 py-2.5 bg-[#111827] text-white rounded-lg text-sm font-medium hover:bg-[#8B1E28] transition-colors disabled:opacity-70 disabled:cursor-not-allowed"
                    >
                      Update Password
                    </button>
                  </div>
                </form>
              </div>
            </div>

            {/* Preferences Section */}
            <div className="bg-white rounded-xl border border-[#E5E2D9] shadow-sm overflow-hidden">
              <div className="p-6 border-b border-[#E5E2D9]">
                <h2 className="font-serif text-xl font-semibold text-[#111827]">Preferences</h2>
              </div>
              
              <div className="p-6 space-y-6">
                <div>
                  <h3 className="text-sm font-medium text-[#111827] mb-4">Appearance</h3>
                  <div className="flex items-center gap-4">
                    <button 
                      onClick={() => setTheme('light')}
                      className={`px-4 py-2 rounded-lg border ${theme === 'light' ? 'border-[#8B1E28] bg-[#8B1E28]/5 text-[#8B1E28]' : 'border-[#E5E2D9] text-[#6B7280] hover:bg-[#F7F6F2]'} font-medium text-sm transition-all`}
                    >
                      Light Mode
                    </button>
                    <button 
                      onClick={() => setTheme('dark')}
                      className={`px-4 py-2 rounded-lg border ${theme === 'dark' ? 'border-[#8B1E28] bg-[#8B1E28]/5 text-[#8B1E28]' : 'border-[#E5E2D9] text-[#6B7280] hover:bg-[#F7F6F2]'} font-medium text-sm transition-all`}
                    >
                      Dark Mode
                    </button>
                    <button 
                      onClick={() => setTheme('system')}
                      className={`px-4 py-2 rounded-lg border ${theme === 'system' ? 'border-[#8B1E28] bg-[#8B1E28]/5 text-[#8B1E28]' : 'border-[#E5E2D9] text-[#6B7280] hover:bg-[#F7F6F2]'} font-medium text-sm transition-all`}
                    >
                      System
                    </button>
                  </div>
                </div>

                <div className="pt-6 border-t border-[#E5E2D9]">
                  <h3 className="text-sm font-medium text-[#111827] mb-4">Email Notifications</h3>
                  <div className="space-y-4">
                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <p className="text-sm font-medium text-[#374151]">Security Alerts</p>
                        <p className="text-xs text-[#6B7280]">Get notified about important security updates.</p>
                      </div>
                      <div className="relative">
                        <input 
                          type="checkbox" 
                          className="sr-only" 
                          checked={notifications.emailAlerts}
                          onChange={() => setNotifications(prev => ({ ...prev, emailAlerts: !prev.emailAlerts }))}
                        />
                        <div className={`block w-10 h-6 rounded-full transition-colors ${notifications.emailAlerts ? 'bg-[#8B1E28]' : 'bg-[#E5E2D9]'}`}></div>
                        <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${notifications.emailAlerts ? 'transform translate-x-4' : ''}`}></div>
                      </div>
                    </label>
                    
                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <p className="text-sm font-medium text-[#374151]">Case Updates</p>
                        <p className="text-xs text-[#6B7280]">Receive summaries of new relevant precedents.</p>
                      </div>
                      <div className="relative">
                        <input 
                          type="checkbox" 
                          className="sr-only" 
                          checked={notifications.caseUpdates}
                          onChange={() => setNotifications(prev => ({ ...prev, caseUpdates: !prev.caseUpdates }))}
                        />
                        <div className={`block w-10 h-6 rounded-full transition-colors ${notifications.caseUpdates ? 'bg-[#8B1E28]' : 'bg-[#E5E2D9]'}`}></div>
                        <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${notifications.caseUpdates ? 'transform translate-x-4' : ''}`}></div>
                      </div>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* Danger Zone */}
            <div className="bg-white rounded-xl border border-red-200 shadow-sm overflow-hidden">
              <div className="p-6">
                <h2 className="font-serif text-xl font-semibold text-red-600 mb-2">Danger Zone</h2>
                <p className="text-sm text-[#6B7280] mb-6">Irreversible actions for your account.</p>
                
                <div className="flex items-center justify-between py-4 border-t border-red-100">
                  <div>
                    <p className="text-sm font-medium text-[#374151]">Sign Out</p>
                    <p className="text-xs text-[#6B7280]">Log out of your current session.</p>
                  </div>
                  <button 
                    onClick={handleLogout}
                    className="px-4 py-2 border border-[#E5E2D9] rounded-lg text-sm font-medium text-[#374151] hover:bg-[#F7F6F2] transition-colors"
                  >
                    Sign Out
                  </button>
                </div>
                
                <div className="flex items-center justify-between py-4 border-t border-red-100">
                  <div>
                    <p className="text-sm font-medium text-[#374151]">Delete Account</p>
                    <p className="text-xs text-[#6B7280]">Permanently delete your account and all data.</p>
                  </div>
                  <button className="px-4 py-2 bg-red-50 text-red-600 border border-red-200 rounded-lg text-sm font-medium hover:bg-red-100 transition-colors">
                    Delete Account
                  </button>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
