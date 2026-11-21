import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { VerifyLocalPage } from './pages/VerifyLocalPage'
import { VerifySocialPage } from './pages/VerifySocialPage'
import { CaseDetailPage } from './pages/CaseDetailPage'
import { VerificationHistoryPage } from './pages/VerificationHistoryPage'
import { SystemAnalyticsPage } from './pages/SystemAnalyticsPage'
import { SystemHealthPage } from './pages/SystemHealthPage'
import { BenchmarkEvaluationView } from './components/BenchmarkEvaluationView'
import { ResearchPaperView } from './components/ResearchPaperView'
import { NotFound } from './components/NotFound'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="verify/local" element={<VerifyLocalPage />} />
        <Route path="verify/social" element={<VerifySocialPage />} />
        <Route path="history" element={<VerificationHistoryPage />} />
        <Route path="analytics" element={<SystemAnalyticsPage />} />
        <Route path="health" element={<SystemHealthPage />} />
        <Route path="case/:caseId" element={<CaseDetailPage />} />
        <Route path="benchmark" element={<BenchmarkEvaluationView />} />
        <Route path="paper" element={<ResearchPaperView />} />
        <Route path="*" element={<NotFound />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
