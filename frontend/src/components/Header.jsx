import React from 'react';
import { Globe, Bot, ChevronDown, ShieldCheck, Sparkles, Zap, BrainCircuit } from 'lucide-react';

export const MODELS = [
  // Built-in Local Neural Engines
  { id: 'aether-neural-local',   name: 'AETHER Frontier Engine (DEEP-Opt)',   provider: 'AETHER Local Neural', badge: 'Flash-Opt',    group: 'builtin' },
  { id: 'aether-speculative',    name: 'AETHER Speculative Fast (0.5B+Draft)',provider: 'AETHER Speculative',  badge: 'Ultra-Fast',   group: 'builtin' },
  { id: 'aether-research-v01',   name: 'AETHER Research MoE-v0.2',            provider: 'AETHER AI Lab',       badge: 'MoE PyTorch',  group: 'builtin' },
  // Ollama local models
  { id: 'ollama:deepseek-r1:7b', name: 'DeepSeek-R1 7B (Reasoning)',          provider: 'Ollama · Local',      badge: 'Reasoning',    group: 'ollama' },
  { id: 'ollama:qwen2.5:7b',    name: 'Qwen2.5 7B (Coding & Tool)',          provider: 'Ollama · Local',      badge: 'Coding',       group: 'ollama' },
  { id: 'ollama:llama3.2:3b',   name: 'Llama 3.2 3B (Fast)',                 provider: 'Ollama · Local',      badge: 'Lightweight',  group: 'ollama' },
];

export default function Header({ 
  selectedModel, 
  onSelectModel, 
  enableWebSearch, 
  onToggleWebSearch,
  enableDeepResearch,
  onToggleDeepResearch,
  enableSelfCorrection,
  onToggleSelfCorrection
}) {
  const currentModel = MODELS.find(m => m.id === selectedModel) || MODELS[0];

  return (
    <header className="h-16 px-6 glass-panel border-b border-slate-800/60 flex items-center justify-between z-10 select-none">
      {/* Left: Model Selector Dropdown */}
      <div className="flex items-center space-x-3">
        <div className="relative group">
          <button className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-700/60 hover:border-cyan-500/50 text-xs font-medium text-white transition-all shadow-sm">
            <Bot className="w-4 h-4 text-cyan-400" />
            <span className="font-outfit font-semibold">{currentModel.name}</span>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>

          {/* Dropdown Menu */}
          <div className="absolute left-0 mt-2 w-80 py-2 glass-panel rounded-xl border border-slate-800 shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto transition-all duration-200 z-50">
            <div className="px-3 py-1 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Active Neural Engines</div>
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
      <div className="flex items-center space-x-2.5">
        {/* Deep Research Toggle Button */}
        <button
          onClick={onToggleDeepResearch}
          title="Autonomous Multi-Hop Web Research & Synthesis"
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${
            enableDeepResearch
              ? 'bg-purple-500/20 border-purple-500/50 text-purple-300 shadow-lg shadow-purple-500/15'
              : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <BrainCircuit className={`w-3.5 h-3.5 ${enableDeepResearch ? 'text-purple-400 animate-pulse' : ''}`} />
          <span>Deep Research</span>
          <span className={`w-1.5 h-1.5 rounded-full ${enableDeepResearch ? 'bg-purple-400' : 'bg-slate-600'}`}></span>
        </button>

        {/* Self-Correction Sandbox Toggle Button */}
        <button
          onClick={onToggleSelfCorrection}
          title="Sandbox Code Execution & Automatic Self-Correction"
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${
            enableSelfCorrection
              ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300 shadow-lg shadow-emerald-500/10'
              : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Zap className="w-3.5 h-3.5 text-emerald-400" />
          <span>Self-Correct</span>
          <span className={`w-1.5 h-1.5 rounded-full ${enableSelfCorrection ? 'bg-emerald-400' : 'bg-slate-600'}`}></span>
        </button>

        {/* Web Search Toggle Button */}
        <button
          onClick={onToggleWebSearch}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${
            enableWebSearch
              ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300 shadow-lg shadow-cyan-500/10'
              : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Globe className={`w-3.5 h-3.5 ${enableWebSearch ? 'text-cyan-400' : ''}`} />
          <span>Search</span>
          <span className={`w-1.5 h-1.5 rounded-full ${enableWebSearch ? 'bg-cyan-400 shadow-glow' : 'bg-slate-600'}`}></span>
        </button>
      </div>
    </header>
  );
}
