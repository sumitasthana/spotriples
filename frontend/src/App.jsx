import { useState, useRef } from 'react'
import axios from 'axios'

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// -------------------------------------------------------------------------
// StatsBar
// -------------------------------------------------------------------------

function StatsBar({ stats, count }) {
  const items = [
    { label: 'Total',      value: count },
    { label: 'Subjects',   value: stats.unique_subjects   ?? 0 },
    { label: 'Predicates', value: stats.unique_predicates ?? 0 },
    { label: 'Objects',    value: stats.unique_objects    ?? 0 },
    { label: 'Negated',    value: stats.negated_count     ?? 0 },
  ]
  return (
    <div className="grid grid-cols-5 gap-3 mb-4">
      {items.map(({ label, value }) => (
        <div key={label} className="bg-blue-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-700">{value}</div>
          <div className="text-xs text-gray-500 mt-1">{label}</div>
        </div>
      ))}
    </div>
  )
}

// -------------------------------------------------------------------------
// ResultsTable — sortable, filterable
// -------------------------------------------------------------------------

function ResultsTable({ relationships }) {
  const [sortCol, setSortCol] = useState(null)
  const [sortDir, setSortDir] = useState('asc')
  const [filterSubj, setFilterSubj] = useState('')
  const [filterPred, setFilterPred] = useState('')
  const [filterObj,  setFilterObj]  = useState('')

  const handleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('asc') }
  }

  let rows = relationships.filter(r =>
    r.subject.toLowerCase().includes(filterSubj.toLowerCase()) &&
    r.predicate.toLowerCase().includes(filterPred.toLowerCase()) &&
    r.object.toLowerCase().includes(filterObj.toLowerCase())
  )

  if (sortCol) {
    rows = [...rows].sort((a, b) => {
      const av = String(a[sortCol] ?? '').toLowerCase()
      const bv = String(b[sortCol] ?? '').toLowerCase()
      return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
    })
  }

  const sortable = ['subject', 'predicate', 'object', 'negated']

  return (
    <div>
      {/* Filter row */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        {[['Subject', filterSubj, setFilterSubj], ['Predicate', filterPred, setFilterPred], ['Object', filterObj, setFilterObj]].map(([label, val, set]) => (
          <input
            key={label}
            placeholder={`Filter ${label}…`}
            value={val}
            onChange={e => set(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
        ))}
      </div>

      <div className="text-xs text-gray-400 mb-1">{rows.length} result{rows.length !== 1 ? 's' : ''}</div>

      <div className="overflow-auto max-h-[480px] border border-gray-200 rounded-lg">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-100 sticky top-0">
            <tr>
              <th className="px-2 py-1 text-left text-gray-400 font-medium w-8">#</th>
              {['subject', 'predicate', 'object', 'negated', 'source_quote'].map(col => (
                <th
                  key={col}
                  onClick={() => sortable.includes(col) && handleSort(col)}
                  className={`px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap ${sortable.includes(col) ? 'cursor-pointer hover:text-blue-600 select-none' : ''}`}
                >
                  {col.replace('_', ' ')}
                  {sortCol === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr
                key={i}
                className={`border-t border-gray-100 ${r.negated ? 'bg-red-50' : i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}`}
              >
                <td className="px-2 py-1.5 text-gray-400 text-xs">{i + 1}</td>
                <td className="px-3 py-2 font-medium text-gray-800">{r.subject}</td>
                <td className="px-3 py-2 text-blue-600 italic">{r.predicate}</td>
                <td className="px-3 py-2 font-medium text-gray-800">{r.object}</td>
                <td className="px-3 py-2 text-center">
                  {r.negated
                    ? <span className="bg-red-100 text-red-700 text-xs px-2 py-0.5 rounded-full">Yes</span>
                    : <span className="bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full">No</span>
                  }
                </td>
                <td className="px-3 py-2 text-gray-500 text-xs max-w-xs truncate" title={r.source_quote}>
                  {r.source_quote}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// -------------------------------------------------------------------------
// AnalyzeTab — bar charts for top subjects / predicates / objects
// -------------------------------------------------------------------------

function AnalyzeTab({ relationships }) {
  if (!relationships.length) {
    return <p className="text-gray-400 text-center py-12">Extract relationships first to see analysis.</p>
  }

  const topN = (key, n = 10) => {
    const counts = {}
    relationships.forEach(r => { counts[r[key]] = (counts[r[key]] || 0) + 1 })
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, n)
  }

  const Bar = ({ label, value, max }) => (
    <div className="flex items-center gap-2 text-sm mb-1.5">
      <span className="w-36 truncate text-gray-600 text-right text-xs shrink-0" title={label}>{label}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-4 relative overflow-hidden">
        <div className="bg-blue-400 h-4 rounded-full transition-all" style={{ width: `${(value / max) * 100}%` }} />
      </div>
      <span className="w-6 text-gray-500 text-xs shrink-0">{value}</span>
    </div>
  )

  const Chart = ({ title, data }) => {
    const max = data[0]?.[1] || 1
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h3 className="font-semibold text-gray-700 mb-3 text-sm">{title}</h3>
        {data.map(([label, value]) => <Bar key={label} label={label} value={value} max={max} />)}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Chart title="Top Subjects"   data={topN('subject')} />
      <Chart title="Top Predicates" data={topN('predicate')} />
      <Chart title="Top Objects"    data={topN('object')} />
    </div>
  )
}

// -------------------------------------------------------------------------
// App — main layout
// -------------------------------------------------------------------------

export default function App() {
  const [tab,             setTab]             = useState('extract')
  const [inputMethod,     setInputMethod]     = useState('paste')
  const [text,            setText]            = useState('')
  const [includeImplicit, setIncludeImplicit] = useState(false)
  const [loading,         setLoading]         = useState(false)
  const [error,           setError]           = useState('')
  const [result,          setResult]          = useState(null)
  const fileRef = useRef()

  // ---- Extract from textarea ----
  const handleExtract = async () => {
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const { data } = await axios.post('/extract', { text, include_implicit: includeImplicit })
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Extraction failed')
    } finally {
      setLoading(false)
    }
  }

  // ---- Extract from uploaded file ----
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('include_implicit', includeImplicit ? 'true' : 'false')
      const { data } = await axios.post('/extract/file', form)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'File extraction failed')
    } finally {
      setLoading(false)
    }
  }

  // ---- Downloads ----
  const downloadCSV = () => {
    if (!result) return
    const cols = ['subject', 'predicate', 'object', 'negated', 'source_quote']
    const header = cols.join(',')
    const rows = result.relationships.map(r =>
      cols.map(c => `"${String(r[c] ?? '').replace(/"/g, '""')}"`).join(',')
    )
    downloadBlob([header, ...rows].join('\n'), 'relationships.csv', 'text/csv')
  }

  const downloadJSON = () => {
    if (!result) return
    downloadBlob(JSON.stringify(result.relationships, null, 2), 'relationships.json', 'application/json')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
        <h1 className="text-xl font-bold text-gray-800">SPO Relationship Extractor</h1>
        <p className="text-sm text-gray-500 mt-0.5">Extract subject–predicate–object triplets from text using OpenAI</p>
      </header>

      <main className="max-w-6xl mx-auto p-6">
        {/* Tabs */}
        <div className="flex gap-1 mb-6 border-b border-gray-200">
          {[['extract', 'Extract'], ['analyze', 'Analyze'], ['guide', 'Guide']].map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${tab === id ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* ---- EXTRACT TAB ---- */}
        {tab === 'extract' && (
          <div className="space-y-4">
            {/* Input method */}
            <div className="flex gap-2">
              {[['paste', 'Paste Text'], ['file', 'Upload File']].map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => setInputMethod(id)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${inputMethod === id ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                >
                  {label}
                </button>
              ))}
            </div>

            {inputMethod === 'paste' ? (
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder="Paste your text here…"
                rows={10}
                className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-y font-mono"
              />
            ) : (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-white">
                <input
                  ref={fileRef}
                  type="file"
                  accept=".txt,.md,.pdf"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                <button
                  onClick={() => fileRef.current?.click()}
                  disabled={loading}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 disabled:opacity-50 text-gray-700 rounded-lg text-sm font-medium"
                >
                  Choose File (.txt, .md, .pdf)
                </button>
                <p className="text-gray-400 text-xs mt-2">File is sent to the API; extraction starts immediately</p>
              </div>
            )}

            {/* Options */}
            <label className="flex items-center gap-2 cursor-pointer select-none text-sm text-gray-700 w-fit">
              <input
                type="checkbox"
                checked={includeImplicit}
                onChange={e => setIncludeImplicit(e.target.checked)}
                className="w-4 h-4 accent-blue-600"
              />
              Include implicit relationships
              <span className="text-xs text-gray-400">(Pass 3 — slower, more noise)</span>
            </label>

            {/* Extract button — paste mode only */}
            {inputMethod === 'paste' && (
              <button
                onClick={handleExtract}
                disabled={loading || !text.trim()}
                className="w-full px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium text-sm transition-colors"
              >
                {loading ? 'Extracting…' : 'Extract Relationships'}
              </button>
            )}

            {loading && (
              <div className="text-center text-sm text-gray-500 animate-pulse">
                Running multi-pass extraction — this may take a moment…
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-sm">
                {error}
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold text-gray-800">
                    Results — {result.count} relationship{result.count !== 1 ? 's' : ''}
                  </h2>
                  <div className="flex gap-2">
                    <button onClick={downloadCSV}  className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium">↓ CSV</button>
                    <button onClick={downloadJSON} className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium">↓ JSON</button>
                  </div>
                </div>
                <StatsBar stats={result.stats} count={result.count} />
                <ResultsTable relationships={result.relationships} />
              </div>
            )}
          </div>
        )}

        {/* ---- ANALYZE TAB ---- */}
        {tab === 'analyze' && (
          <AnalyzeTab relationships={result?.relationships ?? []} />
        )}

        {/* ---- GUIDE TAB ---- */}
        {tab === 'guide' && (
          <div className="max-w-2xl space-y-4 text-sm text-gray-700">
            <h2 className="text-base font-semibold">How it works</h2>
            <dl className="space-y-3">
              <div>
                <dt className="font-medium">Pass 1 — Explicit extraction</dt>
                <dd className="text-gray-500 mt-0.5">Each text chunk is first rewritten by the LLM to replace all pronouns with their antecedents, then explicit SPO relationships are extracted with few-shot guided prompts.</dd>
              </div>
              <div>
                <dt className="font-medium">Pass 2 — Cross-chunk reasoning</dt>
                <dd className="text-gray-500 mt-0.5">Pass 1 results are summarised (token-budgeted) and sent back to the LLM to discover NEW relationships derivable by combining two or more existing ones.</dd>
              </div>
              <div>
                <dt className="font-medium">Pass 3 — Implicit (optional)</dt>
                <dd className="text-gray-500 mt-0.5">Enable the checkbox to also extract strongly implied but unstated relationships. Expect more results but also more noise.</dd>
              </div>
            </dl>
            <h2 className="text-base font-semibold pt-2">Tips</h2>
            <ul className="list-disc pl-5 space-y-1 text-gray-500">
              <li>Structured text (reports, articles, contracts) yields the best results.</li>
              <li>Use the filter boxes in the table to explore specific entities.</li>
              <li>Negated relationships are highlighted in red.</li>
              <li>Download CSV for use in Gephi, Neo4j, or Excel.</li>
            </ul>
          </div>
        )}
      </main>
    </div>
  )
}
