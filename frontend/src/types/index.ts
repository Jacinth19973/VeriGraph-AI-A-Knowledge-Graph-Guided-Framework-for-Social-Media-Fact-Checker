export interface EvidenceItem {
  evidence_id: string
  evidence_type?: string
  retrieval_channel?: string
  provider?: string
  title?: string
  url?: string
  snippet?: string
  domain?: string
  source?: string
  published_at?: string
  relevance_score?: number
  credibility_score?: number
  fusion_score?: number
  cluster_id?: number
  provenance_type?: string
  url_status?: string
  display_url?: string
  is_clickable?: boolean
  display_text?: string
  sample_id?: string
  dataset?: string
  [key: string]: any
}

export interface NarrativeCluster {
  cluster_id: number
  size?: number
  theme?: string
  theme_keywords?: string[]
  source_count?: number
  unique_source_domains?: string[]
  mean_cluster_coherence?: number
  mean_credibility?: number
  mean_relevance?: number
  evidence_sufficiency?: string
  support_signal_count?: number
  conflict_signal_count?: number
  support_ratio?: number | null
  conflict_ratio?: number | null
  representative_evidence_ids?: string[]
  representative_titles?: string[]
  evidence_ids?: string[]
  retrieval_channels?: string[]
  source_diversity?: number
  conflict_summary?: string
  narrative_summary?: string
  [key: string]: any
}

export interface AgentAssessment {
  evidence_id?: string
  relevance?: string
  stance?: string
  quality?: string
  confidence?: number
  reason?: string
  status?: string
  [key: string]: any
}

export interface SpecialistAgent {
  agent?: string
  status?: string
  assessment?: string
  confidence?: number
  overall_assessment?: string
  key_findings?: string[]
  uncertainty_flags?: string[]
  reasoning?: string
  supporting_evidence_ids?: string[]
  refuting_evidence_ids?: string[]
  evidence_assessments?: AgentAssessment[]
  source_assessments?: any[]
  temporal_findings?: any[]
  location_findings?: any[]
  contradiction_evidence_ids?: string[]
  evidence_coverage?: number
  retrieval_support?: number
  composite_assessment_score?: number
  score_components?: Record<string, number>
  [key: string]: any
}

export interface SpecialistsResponse {
  evidence_verification?: SpecialistAgent
  claim_verification?: SpecialistAgent
  source_credibility?: SpecialistAgent
  temporal_location_consistency?: SpecialistAgent
  specialist_summary?: Record<string, any>
  evidence_assessment_matrix?: any[]
  [key: string]: any
}

export interface DebateTurn {
  round?: number
  speaker?: string
  position?: string
  confidence?: number
  arguments?: any[]
  response_to_other?: any[]
  key_summary?: string
  key_evidence_summary?: string
  uncertainty_concessions?: string[]
  [key: string]: any
}

export interface DebateResponse {
  max_rounds?: number
  actual_rounds?: number
  stop_reason?: string
  turns?: DebateTurn[]
  debate_log?: any[]
  debate_history?: any[]
  specialist_summary?: Record<string, any>
  uncertainty_profile?: Record<string, any>
  judge?: Record<string, any>
  final_verdict?: string
  final_confidence?: number
  [key: string]: any
}

export interface KGContext {
  status?: string
  entities?: string[]
  persons?: string[]
  organizations?: string[]
  locations?: string[]
  events?: string[]
  dates?: string[]
  sources?: string[]
  source_domains?: string[]
  relationships?: any[]
  related_claims?: string[]
  nodes_inspected?: number
  nodes_returned?: number
  relationships_returned?: number
  duration_seconds?: number
  neo4j_used?: boolean
  reason?: string
  [key: string]: any
}

export interface KGCandidate {
  candidate_id?: string
  entity_type?: string
  name?: string
  properties?: Record<string, any>
  source_evidence_ids?: string[]
  source_case_id?: string
  confidence?: number
  validation_status?: string
  source_quote?: string
  [key: string]: any
}

export interface KGResponse {
  kg_context?: KGContext
  kg_expansion?: Record<string, any>
  kg_findings?: string[]
  kg_gaps?: string[]
  expansion_candidates?: KGCandidate[]
  [key: string]: any
}

export interface VerificationFactor {
  factor?: string
  assessment?: string
  confidence?: number
  status?: string
  detail?: string
  [key: string]: any
}

export interface ExplanationResponse {
  summary?: string
  verdict_reason?: string
  verification_factors?: VerificationFactor[]
  uncertainty?: Record<string, any>
  supporting_evidence?: string[]
  contradicting_evidence?: string[]
  neutral_evidence?: string[]
  temporal_findings?: string[]
  location_findings?: string[]
  source_findings?: string[]
  kg_findings?: string[]
  debate_findings?: string[]
  uncertainty_factors?: string[]
  knowledge_gaps?: string[]
  natural_language_summary?: string
  confidence_definition?: string
  provenance_chain?: any[]
  agent_summary?: any[]
  [key: string]: any
}

export interface AuditStage {
  stage: string
  status: string
  start_time?: string
  end_time?: string
  duration_seconds?: number
  message?: string
  error?: string
  [key: string]: any
}

export interface AuditResponse {
  case_id: string
  created_at?: string
  pipeline_started_at?: string
  pipeline_completed_at?: string
  total_duration_seconds?: number
  stages?: AuditStage[]
  processing_metrics?: Record<string, any>
  [key: string]: any
}

export interface CaseSummary {
  case_id: string
  claim: string
  verdict: string
  confidence: number
  created_at: string
  source_platform?: string | null
  source_url?: string | null
  processing_time_seconds?: number
  total_evidence?: number
  narrative_count?: number
  [key: string]: any
}

export interface CaseListResponse {
  cases: CaseSummary[]
  total: number
  page?: number
  page_size?: number
}

export interface VerificationRequest {
  claim: string
  image_path?: string
  promote_knowledge?: boolean
  case_id?: string
  metadata?: Record<string, any>
}

export interface VerificationJobResponse {
  job_id: string
  case_id: string
  status: string
  created_at?: string
  case_input?: VerificationRequest
  current_stage?: string
  stage_status?: string
  result?: CaseDetail
  [key: string]: any
}

export interface CaseDetail {
  case_id: string
  claim: string
  image_path?: string
  created_at?: string
  input_metadata?: Record<string, any>
  retrieval?: Record<string, any>
  narratives?: {
    clusters?: NarrativeCluster[]
    fused_evidence?: EvidenceItem[]
    overall_evidence_sufficiency?: string
    global_cluster_coherence?: number
    global_credibility?: number
    diagnostics?: Record<string, any>
    [key: string]: any
  }
  specialists?: SpecialistsResponse
  uncertainty?: Record<string, any>
  debate?: DebateResponse
  judge?: Record<string, any>
  explanation?: ExplanationResponse
  kg?: KGResponse
  kg_update?: Record<string, any>
  audit?: AuditStage[]
  processing_metrics?: Record<string, any>
  [key: string]: any
}

export interface SocialMediaPost {
  platform: string
  post_url: string
  canonical_url?: string
  post_id?: string
  author?: string
  author_id?: string
  caption?: string
  full_text?: string
  image_url?: string
  image_urls?: string[]
  thumbnail_url?: string
  video_url?: string
  published_at?: string
  location?: string
  hashtags?: string[]
  extraction_method?: string
  extraction_status?: string
  access_status?: string
  error?: string
  error_message?: string
  metadata?: Record<string, any>
  [key: string]: any
}

export interface SocialMediaVerificationRequest {
  platform: string
  post_url: string
  use_extracted_content: boolean
  claim?: string
  promote_knowledge?: boolean
}

export interface EvidenceListResponse {
  evidence: EvidenceItem[]
  total: number
  [key: string]: any
}

export interface AnalyticsResponse {
  total_cases: number
  verdict_distribution: Record<string, number>
  average_processing_time: number
  average_debate_rounds: number
  agent_agreement_rate: number
  average_evidence_count: number
  average_narrative_count: number
  source_distribution: Record<string, number>
  platform_distribution: Record<string, number>
  verdict_by_platform: Record<string, Record<string, number>>
  [key: string]: any
}

export interface SystemHealthResponse {
  status: string
  timestamp: string
  services: Record<string, {
    name: string
    status: string
    details?: string
    latency_ms?: number
    [key: string]: any
  }>
  [key: string]: any
}
