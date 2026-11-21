import axios, { AxiosError } from 'axios'
import type {
  AnalyticsResponse,
  AuditResponse,
  CaseDetail,
  CaseListResponse,
  CaseSummary,
  DebateResponse,
  EvidenceItem,
  EvidenceListResponse,
  ExplanationResponse,
  KGResponse,
  NarrativeCluster,
  SocialMediaPost,
  SocialMediaVerificationRequest,
  SpecialistsResponse,
  SystemHealthResponse,
  VerificationJobResponse,
  VerificationRequest,
} from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:8000/api').replace(/\/$/, '')

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,  // 60s timeout per request - job polling handles long pipelines
  headers: { 'Content-Type': 'application/json' },
})

export function getApiError(error: unknown, fallback = 'Request failed'): string {
  if (error instanceof AxiosError) {
    const status = error.response?.status
    const detail = error.response?.data?.detail
    const detailMsg = typeof detail === 'string' ? detail : detail && typeof detail === 'object' ? JSON.stringify(detail) : ''
    
    // Differentiate error types for better diagnostics
    if (detailMsg) return detailMsg
    if (status === 404) return 'Resource not found (404)'
    if (status === 500) return 'Backend internal server error (500)'
    if (status === 502 || status === 503) return 'Service unavailable (502/503)'
    if (error.code === 'ECONNABORTED') return 'Verification request timed out'
    if (error.code === 'ERR_NETWORK') return 'Connection refused — is backend running?'
    if (error.message) return error.message
  }
  return error instanceof Error ? error.message : fallback
}

/* All service methods return the response body, not AxiosResponse.
   This keeps page code consistent and prevents response.data/type mismatches. */

export const healthApi = {
  check: async (): Promise<SystemHealthResponse> => (await api.get<SystemHealthResponse>('/health/')).data,
  neo4jStats: async (): Promise<Record<string, any>> => (await api.get('/health/neo4j/stats')).data,
}

export const verificationApi = {
  start: async (request: VerificationRequest): Promise<VerificationJobResponse> => {
    const response = await api.post<VerificationJobResponse>('/verification/start', request)
    const data = response.data
    
    // Extract job_id from message field if not at top level
    // Backend format: "Verification job started. Job ID: <job_id>"
    let jobId = data.job_id
    if (!jobId && data.message) {
      const match = data.message.match(/Job ID:\s*(\S+)/)
      if (match) {
        jobId = match[1]
      }
    }
    
    // Return enhanced response with derived job_id
    return {
      ...data,
      job_id: jobId,
    } as VerificationJobResponse
  },

  status: async (jobId: string): Promise<VerificationJobResponse> =>
    (await api.get<VerificationJobResponse>(`/verification/status/${encodeURIComponent(jobId)}`)).data,

  result: async (caseId: string): Promise<CaseDetail> =>
    (await api.get<CaseDetail>(`/verification/result/${encodeURIComponent(caseId)}`)).data,

  sync: async (request: VerificationRequest): Promise<CaseDetail> =>
    (await api.post<CaseDetail>('/verification/sync', request)).data,

  poll: async (jobId: string, maxAttempts = 180, intervalMs = 2000): Promise<VerificationJobResponse> => {
    let attempts = 0
    while (attempts < maxAttempts) {
      await new Promise(r => setTimeout(r, intervalMs))
      const status = await api.get<VerificationJobResponse>(
        `/verification/status/${encodeURIComponent(jobId)}`
      ).then(res => res.data)
      if (status.status === 'completed' || status.status === 'failed') {
        return status
      }
      attempts++
    }
    return await api.get<VerificationJobResponse>(
      `/verification/status/${encodeURIComponent(jobId)}`
    ).then(res => res.data)
  },
}

export const casesApi = {
  list: async (params?: {
    page?: number
    page_size?: number
    verdict?: string
    platform?: string
    search?: string
  }): Promise<CaseListResponse> => (await api.get<CaseListResponse>('/cases/', { params })).data,

  get: async (caseId: string): Promise<CaseDetail> =>
    (await api.get<CaseDetail>(`/cases/${encodeURIComponent(caseId)}`)).data,

  summary: async (caseId: string): Promise<CaseSummary> =>
    (await api.get<CaseSummary>(`/cases/${encodeURIComponent(caseId)}/summary`)).data,

  delete: async (caseId: string): Promise<void> => {
    await api.delete(`/cases/${encodeURIComponent(caseId)}`)
  },
}

export const evidenceApi = {
  get: async (caseId: string, params?: {
    channel?: string
    provider?: string
    min_relevance?: number
    min_credibility?: number
    min_fusion?: number
    cluster_id?: number
    provenance_type?: string
  }): Promise<EvidenceListResponse> =>
    (await api.get<EvidenceListResponse>(`/evidence/${encodeURIComponent(caseId)}`, { params })).data,

  detail: async (caseId: string, evidenceId: string): Promise<EvidenceItem> =>
    (await api.get<EvidenceItem>(`/evidence/${encodeURIComponent(caseId)}/${encodeURIComponent(evidenceId)}`)).data,

  summary: async (caseId: string): Promise<Record<string, any>> =>
    (await api.get(`/evidence/${encodeURIComponent(caseId)}/stats/summary`)).data,
}

export const narrativesApi = {
  get: async (caseId: string): Promise<Record<string, any>> =>
    (await api.get(`/narratives/${encodeURIComponent(caseId)}`)).data,

  detail: async (caseId: string, clusterId: number): Promise<NarrativeCluster> =>
    (await api.get<NarrativeCluster>(`/narratives/${encodeURIComponent(caseId)}/${clusterId}`)).data,
}

export const specialistsApi = {
  get: async (caseId: string): Promise<SpecialistsResponse> =>
    (await api.get<SpecialistsResponse>(`/agents/${encodeURIComponent(caseId)}`)).data,

  detail: async (caseId: string, agentName: string): Promise<Record<string, any>> =>
    (await api.get(`/agents/${encodeURIComponent(caseId)}/${encodeURIComponent(agentName)}`)).data,
}

export const debateApi = {
  get: async (caseId: string): Promise<DebateResponse> =>
    (await api.get<DebateResponse>(`/debate/${encodeURIComponent(caseId)}`)).data,

  promptTrace: async (caseId: string): Promise<Record<string, any>> =>
    (await api.get(`/debate/${encodeURIComponent(caseId)}/prompt-trace`)).data,
}

export const kgApi = {
  get: async (caseId: string): Promise<KGResponse> =>
    (await api.get<KGResponse>(`/knowledge-graph/${encodeURIComponent(caseId)}`)).data,

  graph: async (caseId: string): Promise<Record<string, any>> =>
    (await api.get(`/knowledge-graph/${encodeURIComponent(caseId)}/graph`)).data,

  stats: async (): Promise<Record<string, any>> => (await api.get('/knowledge-graph/stats')).data,
}

export const explanationApi = {
  get: async (caseId: string): Promise<ExplanationResponse> =>
    (await api.get<ExplanationResponse>(`/explanation/${encodeURIComponent(caseId)}`)).data,

  naturalLanguage: async (caseId: string): Promise<Record<string, any>> =>
    (await api.get(`/explanation/${encodeURIComponent(caseId)}/natural-language`)).data,
}

export const auditApi = {
  get: async (caseId: string): Promise<AuditResponse> =>
    (await api.get<AuditResponse>(`/audit/${encodeURIComponent(caseId)}`)).data,

  timeline: async (caseId: string): Promise<Record<string, any>> =>
    (await api.get(`/audit/${encodeURIComponent(caseId)}/timeline`)).data,
}

export const analyticsApi = {
  overview: async (): Promise<AnalyticsResponse> =>
    (await api.get<AnalyticsResponse>('/analytics/overview')).data,

  verdictTrends: async (days = 30): Promise<Record<string, any>> =>
    (await api.get('/analytics/verdict-trends', { params: { days } })).data,

  sourceQuality: async (): Promise<Record<string, any>> =>
    (await api.get('/analytics/source-quality')).data,

  evidenceStats: async (): Promise<Record<string, any>> =>
    (await api.get('/analytics/evidence-stats')).data,
}

export const socialMediaApi = {
  platforms: async (): Promise<Record<string, any>> =>
    (await api.get('/social-media/platforms')).data,

  fetch: async (request: { platform: string; post_url: string }): Promise<SocialMediaPost> =>
    (await api.post<SocialMediaPost>('/social-media/fetch', request)).data,

  verify: async (request: SocialMediaVerificationRequest): Promise<CaseDetail> =>
    (await api.post<CaseDetail>('/social-media/verify', request)).data,
}

export default api
