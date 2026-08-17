import React, { useRef, useState } from 'react';
import { Send, Square, Paperclip, Mic, FileText, X, Sparkles } from 'lucide-react';

export default function ChatInput({ 
  onSend, 
  isGenerating, 
  onStop,
  attachedFiles,
  onFileUpload,
  onRemoveFile
}) {
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const fileInputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((!input.trim() && attachedFiles.length === 0) || isGenerating) return;
    onSend(input);
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onFileUpload(files);
    }
  };

  return (
    <div className="p-4 glass-panel border-t border-slate-800/60">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto space-y-3">
        {/* Attached Files Chips Bar */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 px-2 py-1">
            {attachedFiles.map((file) => (
              <div 
                key={file.id} 
                className="flex items-center space-x-2 px-3 py-1 rounded-xl bg-slate-800 border border-slate-700 text-xs text-slate-200 animate-fade-in"
              >
                <FileText className="w-3.5 h-3.5 text-cyan-400" />
                <span className="truncate max-w-[150px]">{file.name}</span>
                <span className="text-[10px] text-slate-400 font-mono">({file.chunks} chunks)</span>
                <button
                  type="button"
                  onClick={() => onRemoveFile(file.id)}
                  className="p-0.5 hover:text-red-400 transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input Text Box with Tool Actions */}
        <div className="relative glass-input rounded-2xl p-2 flex items-end space-x-2 shadow-2xl">
          {/* File Attachment Button */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            multiple
            className="hidden"
            accept=".pdf,.docx,.txt,.csv,.json,.md"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2.5 rounded-xl hover:bg-slate-800/80 text-slate-400 hover:text-cyan-400 transition-colors"
            title="Attach PDF, DOCX, TXT, CSV documents"
          >
            <Paperclip className="w-4 h-4" />
          </button>

          {/* Voice Record Toggle */}
          <button
            type="button"
            onClick={() => setIsRecording(!isRecording)}
            className={`p-2.5 rounded-xl transition-colors ${
              isRecording 
                ? 'bg-red-500/20 text-red-400 animate-pulse' 
                : 'hover:bg-slate-800/80 text-slate-400 hover:text-slate-200'
            }`}
            title="Voice input dictation"
          >
            <Mic className="w-4 h-4" />
          </button>

          {/* Main Textarea */}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isRecording ? "Listening to voice..." : "Ask anything or describe a task... (Shift + Enter for line break)"}
            rows={1}
            className="flex-1 bg-transparent border-none text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none max-h-36 py-2 px-1 font-normal"
          />

          {/* Submit / Stop Button */}
          {isGenerating ? (
            <button
              type="button"
              onClick={onStop}
              className="p-2.5 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 transition-all flex items-center justify-center shadow-lg shadow-red-500/10"
              title="Stop Generation"
            >
              <Square className="w-4 h-4 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim() && attachedFiles.length === 0}
              className="p-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 disabled:opacity-40 disabled:hover:from-cyan-500 disabled:hover:to-indigo-600 text-white font-medium shadow-lg shadow-cyan-500/20 transition-all active:scale-[0.96]"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Sub-label footer */}
        <div className="flex items-center justify-between text-[11px] text-slate-500 px-2">
          <span>AI Operating Layer with Model Router & Web RAG</span>
          <span>LaTeX Math $...$ & $$...$$ supported</span>
        </div>
      </form>
    </div>
  );
}
