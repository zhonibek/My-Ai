import React from 'react';
import { Globe, Bot, ChevronDown, ShieldCheck } from 'lucide-react';

export const MODELS = [
  // Built-in Local Neural Engines
  { id: 'aether-neural-local',   name: 'AETHER Neural Engine (RoPE-SwiGLU)', provider: 'AETHER Local Neural', badge: 'Local Neural', group: 'builtin' },
  { id: 'aether-research-v01',   name: 'AETHER Research Model-v0.1',         provider: 'AETHER AI Research',  badge: 'Custom PyTorch', group: 'builtin' },
  // Ollama local models
  { id: 'ollama:deepseek-r1:7b', name: 'DeepSeek-R1 7B',                    provider: 'Ollama · Local',      badge: 'Reasoning',    group: 'ollama' },
  { id: 'ollama:qwen2.5:7b',    name: 'Qwen2.5 7B',                         provider: 'Ollama · Local',      badge: 'Coding',       group: 'ollama' },
  { id: 'ollama:llama3.2:3b',   name: 'Llama 3.2 3B',                       provider: 'Ollama · Local',      badge: 'Fast',         group: 'ollama' },
  { id: 'ollama:phi3.5',        name: 'Phi-3.5 Mini',                       provider: 'Ollama · Local',      badge: 'Compact',      group: 'ollama' },
];

export default function Header({ 
  selectedModel, 
  onSelectModel, 
  enableWebSearch, 
  onToggleWebSearch 
}) {
  const currentModel = MODELS.find(m => m.id === selectedModel) || MODELS[0];

  return (
    <header className="h-16 px-6 glass-panel border-b border-slate-800/60 flex items-center justify-between z-10">
      {/* Left: Model Selector Dropdown */}
      <div className="flex items-center space-x-4">
        <div className="relative group">
          <button className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-700/60 hover:border-cyan-500/50 text-xs font-medium text-white transition-all shadow-sm">
            <Bot className="w-4 h-4 text-cyan-400" />
            <span className="font-outfit font-semibold">{currentModel.name}</span>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>

          {/* Dropdown Menu */}
          <div className="absolute left-0 mt-2 w-80 py-2 glass-panel rounded-xl border border-slate-800 shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto transition-all duration-200 z-50">
            <div className="px-3 py-1 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Local AI Engines</div>
            {MODELS.map((m) => (
              <button
                key={m.id}
                onClick={() => onSelectModel(m.id)}
                className={`w-full px-3 py-2 text-left text-xs flex items-center justify-between hover:bg-slate-800/80 transition-colors ${selectedModel === m.id ? 'bg-cyan-500/10 text-cyan-400 font-medium' : 'text-slate-300'}`}
              >
                <div>
                  <div className="font-medium">{m.name}</div>
                  <div className="text-[10px] text-slate-500">{m.provider}</div>
                </div>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono">
                  {m.badge}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Right: Controls & Toggles */}
      <div className="flex items-center space-x-3">
        {/* Web Search Toggle Button */}
        <button
          onClick={onToggleWebSearch}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${
            enableWebSearch
              ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300 shadow-lg shadow-cyan-500/10'
              : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Globe className={`w-3.5 h-3.5 ${enableWebSearch ? 'text-cyan-400 animate-spin-slow' : ''}`} />
          <span>Web Search</span>
          <span className={`w-2 h-2 rounded-full ${enableWebSearch ? 'bg-cyan-400 shadow-glow' : 'bg-slate-600'}`}></span>
        </button>

        {/* Security Badge */}
        <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span className="text-[11px] font-medium">100% Local & Proprietary Engine</span>
        </div>
      </div>
    </header>
  );
}
