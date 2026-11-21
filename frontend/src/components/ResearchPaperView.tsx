import React, { useState } from 'react';
import {
  FileText,
  Copy,
  Check,
  Download,
  BookOpen,
  Code2,
  ExternalLink,
  Layers,
  Scale,
  Shield,
  CheckCircle2,
  AlertCircle,
  Database,
  Cpu,
  Clock,
  Sparkles
} from 'lucide-react';

interface Props {
  onSwitchTab?: (tab: string) => void;
}

import { LATEX_PAPER_CODE } from '../assets/paperCode';
export { LATEX_PAPER_CODE };

export const ResearchPaperView: React.FC<Props> = ({ onSwitchTab }) => {
  const [activeSubView, setActiveSubView] = useState<'interactive' | 'latex'>('interactive');
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(LATEX_PAPER_CODE);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleDownloadTex = () => {
    const blob = new Blob([LATEX_PAPER_CODE], { type: 'application/x-tex;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'verigraph_ai_paper.tex';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => window.URL.revokeObjectURL(url), 10000);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
    <div className="max-w-7xl mx-auto space-y-6">
      {/* HEADER BANNER */}
      <div className="bg-white border border-indigo-200 rounded-2xl p-6 shadow-sm relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-50 rounded-full blur-3xl pointer-events-none" />
        
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 relative z-10">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[11px] font-bold tracking-wide uppercase px-2.5 py-1 rounded-md bg-indigo-100 text-indigo-700 border border-indigo-200">
                IEEE Conference Paper Format (LaTeX .tex)
              </span>
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200">
                Peer-Review Ready
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-gray-900 tracking-tight">
              VeriGraph AI: An Evidence-Grounded, Knowledge-Graph-Assisted Multi-Agent Framework for Social Media Fact-Checking
            </h1>
            <p className="text-xs sm:text-sm text-gray-600 mt-1">
              Department of Artificial Intelligence and Data Science · Mepco Schlenk Engineering College, Sivakasi, Tamil Nadu, India
            </p>
          </div>

          <div className="flex items-center gap-2.5 shrink-0">
            <button
              onClick={handleCopy}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-xl shadow-lg transition"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-700" /> : <Copy className="w-4 h-4" />}
              <span>{copied ? 'Copied Full LaTeX!' : 'Copy LaTeX Source'}</span>
            </button>

            <button
              onClick={handleDownloadTex}
              className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-700 border border-gray-300 text-xs font-semibold px-4 py-2 rounded-xl transition"
              title="Download LaTeX source file"
            >
              <Download className="w-4 h-4 text-cyan-400" />
              <span>Download .tex</span>
            </button>
          </div>
        </div>

        {/* SUBVIEW TOGGLES */}
        <div className="flex items-center gap-2 mt-6 pt-4 border-t border-gray-200">
          <button
            onClick={() => setActiveSubView('interactive')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition ${
              activeSubView === 'interactive'
                ? 'bg-indigo-100 text-indigo-700 border border-indigo-300'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Interactive Paper Explorer & Visual Breakdown
          </button>

          <button
            onClick={() => setActiveSubView('latex')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition ${
              activeSubView === 'latex'
                ? 'bg-indigo-100 text-indigo-700 border border-indigo-300'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            Complete LaTeX Code (.tex)
          </button>
        </div>
      </div>

      {/* METRIC HIGHLIGHTS CARDS */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-white border border-gray-200 rounded-xl p-3.5 text-center">
          <div className="text-xs text-gray-500 uppercase font-medium">Verdict Accuracy</div>
          <div className="text-xl font-bold text-emerald-700 mt-1">86.0%</div>
          <div className="text-[10px] text-emerald-600 mt-0.5">+22% over vanilla RAG</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-xl p-3.5 text-center">
          <div className="text-xs text-gray-500 uppercase font-medium">Macro-F1 Score</div>
          <div className="text-xl font-bold text-sky-700 mt-1">0.88</div>
          <div className="text-[10px] text-sky-600 mt-0.5">High precision across 4 classes</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-xl p-3.5 text-center">
          <div className="text-xs text-gray-500 uppercase font-medium">Score Margin (Δ)</div>
          <div className="text-xl font-bold text-indigo-700 mt-1">0.48</div>
          <div className="text-[10px] text-indigo-600 mt-0.5">Fixed flatline (was 0.04)</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-xl p-3.5 text-center">
          <div className="text-xs text-gray-500 uppercase font-medium">Retrieval nDCG@5</div>
          <div className="text-xl font-bold text-violet-700 mt-1">0.83</div>
          <div className="text-[10px] text-violet-600 mt-0.5">Dual FAISS BGE+CLIP</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-xl p-3.5 text-center">
          <div className="text-xs text-gray-500 uppercase font-medium">Grounding Rate</div>
          <div className="text-xl font-bold text-amber-700 mt-1">92.4%</div>
          <div className="text-[10px] text-amber-600 mt-0.5">Strict citation constraints</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-xl p-3.5 text-center">
          <div className="text-xs text-gray-500 uppercase font-medium">End-to-End Latency</div>
          <div className="text-xl font-bold text-cyan-400 mt-1">7.42s</div>
          <div className="text-[10px] text-cyan-500/80 mt-0.5">4GB GPU + CPU fallback</div>
        </div>
      </div>

      {/* VIEW: RAW LATEX CODE */}
      {activeSubView === 'latex' && (
        <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-2xl">
          <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Code2 className="w-4 h-4 text-indigo-700" />
              <span className="text-xs font-mono font-medium text-gray-600">verigraph_ai_paper.tex</span>
              <span className="text-[10px] bg-gray-100 text-gray-500 px-2 py-0.5 rounded">
                {LATEX_PAPER_CODE.split('\n').length} lines · ~40 KB
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 text-xs text-indigo-700 hover:text-indigo-700 bg-indigo-50 hover:bg-indigo-100 px-3 py-1 rounded-lg transition"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-700" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
              <a
                href="/verigraph_ai_paper.tex"
                download="verigraph_ai_paper.tex"
                className="flex items-center gap-1.5 text-xs text-gray-600 hover:text-gray-900 bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-lg transition"
              >
                <Download className="w-3.5 h-3.5 text-cyan-400" />
                <span>Save</span>
              </a>
            </div>
          </div>
          <pre className="p-4 text-xs font-mono text-gray-600 bg-gray-50 overflow-x-auto max-h-[750px] leading-relaxed selection:bg-indigo-600 selection:text-white">
            {LATEX_PAPER_CODE}
          </pre>
        </div>
      )}

      {/* VIEW: INTERACTIVE PAPER EXPLORER */}
      {activeSubView === 'interactive' && (
        <div className="space-y-6">
          {/* ABSTRACT SECTION */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-3">
            <div className="flex items-center gap-2 text-indigo-700 font-semibold text-sm">
              <FileText className="w-4 h-4" />
              <span>Abstract Summary</span>
            </div>
            <p className="text-sm text-gray-600 leading-relaxed text-justify">
              Social media claims increasingly combine multimodal text, manipulated imagery, complex temporal markers, and cross-source reporting, making reliable automated verification difficult. When retrieval pipelines rely on unanchored semantic similarity, retrieved sources often share topically related keywords while missing direct relevance, yielding flatline similarity distributions and misleading justifications. Furthermore, multi-agent reasoning frameworks frequently collapse into premature "insufficient evidence" verdicts whenever local language models encounter parsing anomalies or unaligned evidence packets. This paper presents <strong className="text-gray-900">VeriGraph AI</strong>, a robust, end-to-end fact-checking framework that integrates multimodal retrieval, claim-anchored evidence scoring, semantic narrative fusion, knowledge-graph expansion, and adversarial multi-agent debate with a deterministic consensus arbiter.
            </p>
          </div>

          {/* VISUAL ARCHITECTURE PREVIEW (CORRESPONDING TO TIKZ FIG. 1) */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-indigo-700 font-semibold text-sm">
                <Layers className="w-4 h-4" />
                <span>Fig. 1. Implemented System Architecture Workflow (TikZ Model)</span>
              </div>
              <span className="text-xs text-gray-500">11 Integrated Pipeline Stages</span>
            </div>

            {/* FLOW DIAGRAM RENDERING */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
              {/* TIER 1 */}
              <div className="bg-gray-50 border border-gray-200 p-4 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-amber-700 font-bold uppercase tracking-wider text-[10px]">
                  <span>Tier 1: Intake</span>
                  <Sparkles className="w-3.5 h-3.5" />
                </div>
                <div className="font-semibold text-gray-700">Claim Decomposition</div>
                <p className="text-gray-500 text-[11px] leading-relaxed">
                  Extracts atomic claims, named entities <code className="text-gray-600">E</code>, temporal years <code className="text-gray-600">T</code>, and locations <code className="text-gray-600">L</code>.
                </p>
                <div className="bg-white p-2 rounded border border-gray-200 text-[10px] text-gray-500">
                  Generates 6 complementary query families (direct, temporal, source, support & refute).
                </div>
              </div>

              {/* TIER 2 */}
              <div className="bg-gray-50 border border-gray-200 p-4 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-sky-700 font-bold uppercase tracking-wider text-[10px]">
                  <span>Tier 2: Retrieval</span>
                  <Database className="w-3.5 h-3.5" />
                </div>
                <div className="font-semibold text-gray-700">Dual-Channel & KG</div>
                <p className="text-gray-500 text-[11px] leading-relaxed">
                  BGE (768-d text) + CLIP (512-d image) in separate FAISS indexes + SerpApi (India gl=in) + Neo4j 2-hop graph paths.
                </p>
                <div className="bg-white p-2 rounded border border-gray-200 text-[10px] text-gray-500">
                  Deduplication by URL hash and normalized title similarity.
                </div>
              </div>

              {/* TIER 3 */}
              <div className="bg-gray-50 border border-gray-200 p-4 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-violet-700 font-bold uppercase tracking-wider text-[10px]">
                  <span>Tier 3: Fusion</span>
                  <Scale className="w-3.5 h-3.5" />
                </div>
                <div className="font-semibold text-gray-700">Claim-Anchored Scoring</div>
                <p className="text-gray-500 text-[11px] leading-relaxed">
                  Calculates <code className="text-gray-600">S(e, C)</code> using 8 weighted dimensions. Agglomerative clustering produces Grounded Narratives with polarity <code className="text-gray-600">P(N_k)</code>.
                </p>
                <div className="bg-white p-2 rounded border border-gray-200 text-[10px] text-emerald-700">
                  Score separation margin Δ = 0.48 (cures flatlining).
                </div>
              </div>

              {/* TIER 4 */}
              <div className="bg-gray-50 border border-gray-200 p-4 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-emerald-700 font-bold uppercase tracking-wider text-[10px]">
                  <span>Tier 4: Adjudication</span>
                  <Shield className="w-3.5 h-3.5" />
                </div>
                <div className="font-semibold text-gray-700">Debate & Arbiter</div>
                <p className="text-gray-500 text-[11px] leading-relaxed">
                  Supporter vs. Skeptic adversarial debate + Judge. Supervised by mathematical consensus fallback to prevent premature Insufficient drops.
                </p>
                <div className="bg-white p-2 rounded border border-gray-200 text-[10px] text-emerald-700">
                  86.0% accuracy with full verifiable audit trail.
                </div>
              </div>
            </div>
          </div>

          {/* MATHEMATICAL FORMULATION SPOTLIGHT */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3">
              <div className="flex items-center gap-2 text-sky-700 font-semibold text-xs uppercase tracking-wider">
                <Scale className="w-4 h-4" />
                <span>Eq. (4) Claim-Anchored Fusion Formulation</span>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl font-mono text-xs text-sky-700 border border-gray-200 overflow-x-auto">
                S(e, C) = 0.22·f_rel + 0.14·f_dir + 0.16·f_ent + 0.14·f_time + 0.08·f_loc + 0.12·f_cred + 0.08·f_nar + 0.06·f_div
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">
                By passing <code className="text-gray-600">claim_text</code> directly into <code className="text-gray-600">_enrich_evidence_item</code> and matching entities and publication years against claim anchors, relevant articles score 0.75–0.92 while out-of-context articles score 0.25–0.38.
              </p>
            </div>

            <div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3">
              <div className="flex items-center gap-2 text-emerald-700 font-semibold text-xs uppercase tracking-wider">
                <Shield className="w-4 h-4" />
                <span>Eq. (12) Deterministic Consensus Arbiter</span>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl font-mono text-xs text-emerald-700 border border-gray-200 overflow-x-auto">
                V_fallback = False if M_ref ≥ 1.4·M_sup ∧ M_ref ≥ 0.65; True if M_sup ≥ 1.4·M_ref ∧ M_sup ≥ 0.65; Misleading if temporal mismatch
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">
                Guarantees that conclusive refutation or support evidence from verified fact-checkers is never discarded when local Ollama models experience transient JSON parsing hitches.
              </p>
            </div>
          </div>

          {/* BENCHMARK EVALUATION TABLES */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-indigo-700 font-semibold text-sm">
                <CheckCircle2 className="w-4 h-4" />
                <span>Table II. Verification Performance Across 100 Benchmark Cases</span>
              </div>
              <span className="text-xs text-gray-500">Ablation & Baseline Comparison</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-500 font-semibold uppercase text-[10px]">
                    <th className="py-2.5 px-3">Framework / Configuration</th>
                    <th className="py-2.5 px-3">Accuracy</th>
                    <th className="py-2.5 px-3">Macro-F1</th>
                    <th className="py-2.5 px-3">Grounding Rate</th>
                    <th className="py-2.5 px-3">Unsupported Claims</th>
                    <th className="py-2.5 px-3">Brier Score</th>
                    <th className="py-2.5 px-3">Avg Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 font-mono">
                  <tr className="hover:bg-gray-50 text-gray-500">
                    <td className="py-2.5 px-3 font-sans text-gray-600">Direct LLM Zero-Shot</td>
                    <td className="py-2.5 px-3">52.0%</td>
                    <td className="py-2.5 px-3">0.48</td>
                    <td className="py-2.5 px-3">N/A</td>
                    <td className="py-2.5 px-3 text-rose-600">38.0%</td>
                    <td className="py-2.5 px-3">0.34</td>
                    <td className="py-2.5 px-3">1.85s</td>
                  </tr>
                  <tr className="hover:bg-gray-50 text-gray-500">
                    <td className="py-2.5 px-3 font-sans text-gray-600">Standard Dense RAG</td>
                    <td className="py-2.5 px-3">64.0%</td>
                    <td className="py-2.5 px-3">0.61</td>
                    <td className="py-2.5 px-3">62.4%</td>
                    <td className="py-2.5 px-3 text-rose-600">24.5%</td>
                    <td className="py-2.5 px-3">0.26</td>
                    <td className="py-2.5 px-3">3.40s</td>
                  </tr>
                  <tr className="hover:bg-gray-50 text-gray-500">
                    <td className="py-2.5 px-3 font-sans text-gray-600">FC-MAD (Debate Only)</td>
                    <td className="py-2.5 px-3">73.0%</td>
                    <td className="py-2.5 px-3">0.72</td>
                    <td className="py-2.5 px-3">71.8%</td>
                    <td className="py-2.5 px-3 text-amber-700">16.2%</td>
                    <td className="py-2.5 px-3">0.21</td>
                    <td className="py-2.5 px-3">8.92s</td>
                  </tr>
                  <tr className="hover:bg-gray-50 text-gray-500">
                    <td className="py-2.5 px-3 font-sans text-gray-600">VeriGraph AI w/o KG</td>
                    <td className="py-2.5 px-3">80.0%</td>
                    <td className="py-2.5 px-3">0.81</td>
                    <td className="py-2.5 px-3">84.5%</td>
                    <td className="py-2.5 px-3 text-emerald-700">8.5%</td>
                    <td className="py-2.5 px-3">0.16</td>
                    <td className="py-2.5 px-3">6.80s</td>
                  </tr>
                  <tr className="hover:bg-gray-50 text-gray-500">
                    <td className="py-2.5 px-3 font-sans text-gray-600">VeriGraph AI w/o Debate</td>
                    <td className="py-2.5 px-3">77.0%</td>
                    <td className="py-2.5 px-3">0.78</td>
                    <td className="py-2.5 px-3">81.2%</td>
                    <td className="py-2.5 px-3 text-amber-700">11.0%</td>
                    <td className="py-2.5 px-3">0.18</td>
                    <td className="py-2.5 px-3">5.15s</td>
                  </tr>
                  <tr className="bg-indigo-50 font-semibold text-gray-900">
                    <td className="py-3 px-3 font-sans text-indigo-700 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-amber-700" />
                      VeriGraph AI (Full Pipeline)
                    </td>
                    <td className="py-3 px-3 text-emerald-700">86.0%</td>
                    <td className="py-3 px-3 text-sky-700">0.88</td>
                    <td className="py-3 px-3 text-emerald-700">92.4%</td>
                    <td className="py-3 px-3 text-emerald-700">3.2%</td>
                    <td className="py-3 px-3 text-emerald-700">0.12</td>
                    <td className="py-3 px-3 text-cyan-400">7.42s</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* HOW TO COMPILE SECTION */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-3">
            <div className="flex items-center gap-2 text-teal-400 font-semibold text-sm">
              <Cpu className="w-4 h-4" />
              <span>Compilation Instructions for Overleaf / TeX Live</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="bg-gray-50 p-4 rounded-xl border border-gray-200 space-y-2">
                <div className="font-semibold text-gray-700">Option 1: Overleaf (Cloud LaTeX)</div>
                <ol className="list-decimal list-inside text-gray-500 space-y-1">
                  <li>Click <strong>Download .tex</strong> above to get <code className="text-gray-600">verigraph_ai_paper.tex</code>.</li>
                  <li>In Overleaf, create a <strong>New Project → Upload Project</strong>.</li>
                  <li>Select compiler <strong>pdfLaTeX</strong> in Menu settings.</li>
                  <li>Click <strong>Recompile</strong> to generate the PDF instantly.</li>
                </ol>
              </div>

              <div className="bg-gray-50 p-4 rounded-xl border border-gray-200 space-y-2">
                <div className="font-semibold text-gray-700">Option 2: Local Terminal (Linux/Mac/Windows)</div>
                <div className="bg-white p-2.5 rounded font-mono text-[11px] text-teal-300 border border-gray-200">
                  pdflatex verigraph_ai_paper.tex<br />
                  bibtex verigraph_ai_paper<br />
                  pdflatex verigraph_ai_paper.tex<br />
                  pdflatex verigraph_ai_paper.tex
                </div>
                <p className="text-gray-500 text-[11px]">
                  Requires <code className="text-gray-600">texlive-latex-extra</code> and <code className="text-gray-600">texlive-fonts-recommended</code>.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
    </div>
  );
};
