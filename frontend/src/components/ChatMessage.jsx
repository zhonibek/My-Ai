import React, { useState, useCallback } from 'react';
import { Bot, User, Copy, Check, ExternalLink, Sparkles, ChevronDown, ChevronUp, Globe, Code } from 'lucide-react';
import { marked } from 'marked';
import katex from 'katex';

// Syntax highlighting for code blocks
const CODE_COLORS = {
  keyword: 'text-violet-400',
  string: 'text-emerald-400',
  comment: 'text-slate-500 italic',
  number: 'text-amber-400',
  fn: 'text-cyan-400',
};

function highlightCode(code, lang) {
  if (!lang || lang === 'text') return escapeHtml(code);
  const escaped = escapeHtml(code);
  return escaped
    .replace(/\/\/.*/g, m => `<span class="text-slate-500 italic">${m}</span>`)
    .replace(/#.*/g, m => `<span class="text-slate-500 italic">${m}</span>`)
    .replace(/\b(def|class|return|import|from|if|else|elif|for|while|try|except|finally|with|as|in|not|and|or|True|False|None|async|await|yield|lambda|pass|break|continue|raise|is|del|global|nonlocal|assert)\b/g,
      m => `<span class="text-violet-400 font-semibold">${m}</span>`)
    .replace(/\b(const|let|var|function|return|import|export|from|if|else|for|while|class|extends|async|await|new|typeof|instanceof|this|default|null|undefined|true|false)\b/g,
      m => `<span class="text-violet-400 font-semibold">${m}</span>`)
    .replace(/(&quot;|&#x27;)(.*?)\1/g, m => `<span class="text-emerald-400">${m}</span>`)
    .replace(/\b\d+\.?\d*\b/g, m => `<span class="text-amber-400">${m}</span>`);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;');
}

// Custom marked renderer with code highlighting
const renderer = new marked.Renderer();
renderer.code = function(code, lang) {
  const highlighted = highlightCode(code, lang);
  const langLabel = lang || 'code';
  return `<div class="code-block-wrapper">
    <div class="code-block-header">
      <span class="code-lang-label">${escapeHtml(langLabel)}</span>
      <button class="copy-code-btn" onclick="navigator.clipboard.writeText(this.closest('.code-block-wrapper').querySelector('code').innerText); this.textContent='Copied!'; setTimeout(()=>this.textContent='Copy',2000)">Copy</button>
    </div>
    <pre class="code-block"><code class="language-${escapeHtml(langLabel)}">${highlighted}</code></pre>
  </div>`;
};

marked.setOptions({ renderer, breaks: true, gfm: true });

export default function ChatMessage({ message }) {
  const [copied, setCopied] = useState(false);
  const [showReasoning, setShowReasoning] = useState(true);
  const [showThinking, setShowThinking] = useState(true);

  const isAssistant = message.sender === 'assistant';

  const handleCopy = useCallback(() => {
    const cleanText = (message.content || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    navigator.clipboard.writeText(cleanText || message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [message.content]);

  // Extract <think>...</think> block
  let thinkingContent = message.thinkingContent || '';
  let finalContent = message.content || '';
  const thinkMatch = finalContent.match(/<think>([\s\S]*?)(?:<\/think>|$)/i);
  if (thinkMatch) {
    thinkingContent = (thinkingContent ? thinkingContent + '\n' : '') + thinkMatch[1].trim();
    finalContent = finalContent.replace(/<think>[\s\S]*?(?:<\/think>|$)/i, '').trim();
  }

  const renderFormattedContent = (content) => {
    if (!content) return '';

    // Block math $$...$$
    let text = content.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
      try { return katex.renderToString(formula, { displayMode: true, throwOnError: false }); }
      catch { return match; }
    });

    // Inline math $...$
    text = text.replace(/\$([^$\n]+?)\$/g, (match, formula) => {
      try { return katex.renderToString(formula, { displayMode: false, throwOnError: false }); }
      catch { return match; }
    });

    try { return marked.parse(text); }
    catch { return text; }
  };

  return (
    <div className={`py-5 px-6 flex space-x-4 transition-colors group ${isAssistant ? 'bg-slate-900/30 border-y border-slate-800/40' : ''}`}>
      {/* Avatar */}
      <div className="shrink-0 mt-0.5">
        {isAssistant ? (
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Bot className="w-4 h-4 text-white" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300">
            <User className="w-4 h-4" />
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden space-y-3 min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center space-x-2 font-medium">
            <span className="text-slate-200">{isAssistant ? 'AETHER' : 'You'}</span>
            {message.modelUsed && (
              <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] text-cyan-400 font-mono">
                {message.modelUsed}
              </span>
            )}
          </div>
          {isAssistant && message.content && (
            <button
              onClick={handleCopy}
              className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 hover:text-slate-200 transition-all text-[11px]"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          )}
        </div>

        {/* Chain-of-Thought Block */}
        {isAssistant && thinkingContent && (
          <div className="rounded-xl border border-indigo-500/30 bg-indigo-950/20 overflow-hidden text-xs">
            <button
              onClick={() => setShowThinking(!showThinking)}
              className="w-full px-3.5 py-2.5 flex items-center justify-between font-medium text-indigo-300 bg-indigo-900/30 hover:bg-indigo-900/50 transition-colors"
            >
              <span className="flex items-center space-x-2">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400 animate-spin-slow" />
                <span className="font-semibold tracking-wide">Ход мыслей AETHER (Chain-of-Thought)</span>
              </span>
              {showThinking ? <ChevronUp className="w-3.5 h-3.5 text-indigo-400" /> : <ChevronDown className="w-3.5 h-3.5 text-indigo-400" />}
            </button>
            {showThinking && (
              <div className="p-3.5 text-[12px] text-indigo-200/80 leading-relaxed font-mono whitespace-pre-wrap bg-slate-950/40 border-t border-indigo-500/20 max-h-60 overflow-y-auto">
                {thinkingContent}
              </div>
            )}
          </div>
        )}

        {/* Orchestrator Reasoning Steps */}
        {isAssistant && message.reasoningSteps && message.reasoningSteps.length > 0 && (
          <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800/80 text-xs">
            <button
              onClick={() => setShowReasoning(!showReasoning)}
              className="w-full flex items-center justify-between font-medium text-slate-300 mb-1"
            >
              <span className="flex items-center space-x-1.5 text-cyan-400">
                <Sparkles className="w-3.5 h-3.5 animate-pulse" />
                <span>Orchestrator Execution Trace</span>
              </span>
              {showReasoning ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            {showReasoning && (
              <div className="mt-2 space-y-1.5 text-[11px] text-slate-400 pl-2 border-l border-cyan-500/30">
                {message.reasoningSteps.map((step, idx) => (
                  <div key={idx} className="flex items-center space-x-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" />
                    <span>{step.explanation || step.status || 'Step processed'}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Source Citations */}
        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-2 py-1">
            {message.sources.map((src, i) => (
              <a
                key={i}
                href={src.url}
                target="_blank"
                rel="noreferrer"
                className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 text-[11px] text-cyan-300 transition-all hover:scale-[1.02]"
              >
                <Globe className="w-3 h-3 text-cyan-400" />
                <span className="truncate max-w-[160px]">{src.title || src.domain}</span>
                <ExternalLink className="w-2.5 h-2.5 opacity-60" />
              </a>
            ))}
          </div>
        )}

        {/* Loading Indicator (streaming) */}
        {isAssistant && !finalContent && !thinkingContent && (
          <div className="flex items-center space-x-1.5 text-slate-500 text-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        )}

        {/* Formatted Content */}
        <div
          className="prose-custom text-sm"
          dangerouslySetInnerHTML={{ __html: renderFormattedContent(finalContent) }}
        />
      </div>
    </div>
  );
}
