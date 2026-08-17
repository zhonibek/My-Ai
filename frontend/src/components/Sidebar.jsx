import React, { useState, useEffect } from 'react';
import {
  Plus, MessageSquare, Trash2, Search, Cpu, Sparkles,
  HardDrive, Zap, Globe, BookOpen, Database
} from 'lucide-react';

function timeAgo(dateVal) {
  try {
    const d = new Date(dateVal);
    if (isNaN(d)) return '';
    const diffMs = Date.now() - d.getTime();
    const mins = Math.floor(diffMs / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  } catch { return ''; }
}

export default function Sidebar({
  conversations,
  activeChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  searchQuery,
  setSearchQuery,
  activeTab,
  setActiveTab
}) {
  const [backendStatus, setBackendStatus] = useState('checking');

  // Poll backend health
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/v1/../', { signal: AbortSignal.timeout(2000) });
        setBackendStatus(res.ok ? 'online' : 'offline');
      } catch {
        setBackendStatus('offline');
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const filteredConversations = conversations.filter(c =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <aside className="w-72 h-full glass-panel flex flex-col border-r border-slate-800/60 select-none">

      {/* Brand Header */}
      <div className="p-4 border-b border-slate-800/60 flex items-center space-x-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 shrink-0">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="font-bold font-outfit text-sm tracking-tight text-white flex items-center gap-1.5">
            AETHER AI
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400 font-mono border border-cyan-500/30">
              v2.0
            </span>
          </h1>
          <p className="text-[10px] text-slate-500 leading-none mt-0.5">Frontier Neural Platform</p>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-3 pb-2">
        <button
          onClick={onNewChat}
          className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-semibold text-sm flex items-center justify-center space-x-2 shadow-lg shadow-cyan-500/20 transition-all duration-200 active:scale-[0.97]"
        >
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Search */}
      <div className="px-3 mb-2">
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
          <input
            type="text"
            placeholder="Search history..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 rounded-lg bg-slate-900/70 border border-slate-800 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/50 focus:bg-slate-900 transition-colors"
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex px-3 space-x-1 mb-2">
        {['chats', 'tools'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-all capitalize ${
              activeTab === tab
                ? 'bg-slate-800 text-cyan-400 shadow-inner'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {tab === 'chats' ? 'Chats' : 'Tools & Info'}
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {activeTab === 'chats' ? (
          filteredConversations.length === 0 ? (
            <div className="text-center py-10 text-xs text-slate-600">
              {searchQuery ? 'No results found' : 'No conversations yet'}
            </div>
          ) : (
            <div className="space-y-0.5">
              {filteredConversations.map(chat => (
                <div
                  key={chat.id}
                  onClick={() => onSelectChat(chat.id)}
                  className={`group relative flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer text-xs font-medium transition-all ${
                    chat.id === activeChatId
                      ? 'bg-gradient-to-r from-slate-800 to-slate-800/60 text-white border border-cyan-500/25 shadow-sm'
                      : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center space-x-2.5 truncate pr-7 min-w-0">
                    <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${chat.id === activeChatId ? 'text-cyan-400' : 'text-slate-600'}`} />
                    <div className="truncate min-w-0">
                      <p className="truncate leading-tight">{chat.title}</p>
                      {chat.updatedAt && (
                        <p className="text-[10px] text-slate-600 group-hover:text-slate-500 mt-0.5">
                          {timeAgo(chat.updatedAt)}
                        </p>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); onDeleteChat(chat.id); }}
                    className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 rounded hover:text-red-400 hover:bg-red-400/10 transition-all"
                    title="Delete"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )
        ) : (
          /* Tools & Info Panel */
          <div className="space-y-2 p-1">
            {[
              { icon: Globe, color: 'text-cyan-400', label: 'Web Search', desc: 'DuckDuckGo + Tavily real-time' },
              { icon: BookOpen, color: 'text-emerald-400', label: 'Document RAG', desc: 'Semantic PDF/text retrieval' },
              { icon: Zap, color: 'text-amber-400', label: 'Code Execution', desc: 'Safe Python sandbox' },
              { icon: Database, color: 'text-indigo-400', label: 'SQLite Memory', desc: 'Persistent chat history' },
            ].map(({ icon: Icon, color, label, desc }) => (
              <div key={label} className="flex items-center space-x-3 px-3 py-2.5 rounded-xl bg-slate-900/50 border border-slate-800/80 text-xs">
                <Icon className={`w-4 h-4 ${color} shrink-0`} />
                <div>
                  <p className="text-slate-200 font-medium">{label}</p>
                  <p className="text-[10px] text-slate-500">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer Status */}
      <div className="p-3 border-t border-slate-800/60 space-y-1.5 text-xs">
        <div className="flex items-center justify-between px-2.5 py-2 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="flex items-center space-x-1.5 text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Backend API</span>
          </span>
          <span className={`text-[10px] font-mono font-bold ${backendStatus === 'online' ? 'text-emerald-400' : 'text-red-400'}`}>
            {backendStatus === 'checking' ? '...' : backendStatus.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center justify-between px-2.5 py-1 text-[10px] text-slate-600">
          <span className="flex items-center space-x-1">
            <HardDrive className="w-3 h-3" />
            <span>Qwen2.5 Local Engine</span>
          </span>
          <span className="text-emerald-500">ACTIVE</span>
        </div>
      </div>
    </aside>
  );
}
