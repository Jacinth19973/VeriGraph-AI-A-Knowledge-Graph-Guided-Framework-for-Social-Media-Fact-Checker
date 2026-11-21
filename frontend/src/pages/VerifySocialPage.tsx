import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CheckCircle, ExternalLink, Globe, Loader2, ShieldAlert } from 'lucide-react'
import { getApiError, socialMediaApi, verificationApi } from '../services/api'
import type { SocialMediaPost } from '../types'

function detectPlatform(url: string): string {
  try {
    const host = new URL(url).hostname.toLowerCase()
    if (host.includes('instagram.com')) return 'instagram'
    if (host.includes('facebook.com') || host.includes('fb.com')) return 'facebook'
    if (host.includes('x.com') || host.includes('twitter.com')) return 'x'
    return 'web'
  } catch {
    return 'web'
  }
}

export function VerifySocialPage() {
  const navigate = useNavigate()
  const [postUrl, setPostUrl] = useState('')
  const [post, setPost] = useState<SocialMediaPost | null>(null)
  const [claim, setClaim] = useState('')
  const [fetching, setFetching] = useState(false)
  const [verifying, setVerifying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState('')

  const fetchPost = async (event: any) => {
    event.preventDefault()
    if (!postUrl.trim()) {
      setError('Please enter a post URL.')
      return
    }

    setFetching(true)
    setError(null)
    setStatus('Extracting public post content...')
    try {
      const result = await socialMediaApi.fetch({
        platform: detectPlatform(postUrl),
        post_url: postUrl.trim(),
      })
      setPost(result)
      setClaim(result.caption || result.full_text || '')
      setStatus('Post extracted')
    } catch (err) {
      setError(getApiError(err, 'Unable to extract this post.'))
      setStatus('')
    } finally {
      setFetching(false)
    }
  }

  const verify = async () => {
    if (!post) return setError('Fetch a post first.')
    if (!claim.trim()) return setError('The extracted caption is empty. Enter the claim manually.')

    setVerifying(true)
    setError(null)
    setStatus('Running verification pipeline...')
    try {
      const jobResult = await verificationApi.start({
        claim: claim.trim(),
        image_path: post.image_url || post.thumbnail_url || undefined,
        promote_knowledge: false,
        metadata: {
          source_platform: post.platform,
          source_url: post.post_url,
          canonical_url: post.canonical_url,
          post_id: post.post_id,
          author: post.author,
          author_id: post.author_id,
          published_at: post.published_at,
          location: post.location,
          extraction_method: post.extraction_method,
          extraction_status: post.extraction_status,
          original_metadata: post.metadata,
        },
      })

      setStatus('Verification in progress. Polling for result...')
      // Poll for completion instead of waiting synchronously
      const final = await verificationApi.poll(jobResult.job_id, (s) =>
        setStatus(`Running: ${s.current_stage || 'queued'}${s.stage_status ? ` (${s.stage_status})` : ''} - this can take 15-20 minutes on a CPU`))
      
      if (final.status === 'failed') {
        throw new Error(final.error || 'Verification job failed.')
      }
      
      if (!final.case_id) throw new Error('The backend did not return a case ID.')
      setStatus('Verification complete')
      window.setTimeout(() => navigate(`/case/${final.case_id}`), 500)
    } catch (err) {
      setError(getApiError(err, 'Social-media verification failed.'))
      setStatus('')
    } finally {
      setVerifying(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <Globe className="w-12 h-12 text-primary-600 mx-auto mb-3" />
          <h1 className="text-3xl font-bold">Verify Social Media Content</h1>
          <p className="text-gray-600 mt-2">Fetch publicly accessible post metadata, review it, then verify the extracted claim.</p>
        </div>

        {!post ? (
          <form onSubmit={fetchPost} className="card space-y-5">
            <div>
              <label htmlFor="post-url" className="block text-sm font-medium mb-2">Post URL</label>
              <input
                id="post-url"
                type="url"
                required
                value={postUrl}
                onChange={(event: { target: { value: string } }) => setPostUrl(event.target.value)}
                disabled={fetching}
                className="input-field"
                placeholder="https://www.instagram.com/... or https://x.com/..."
              />
            </div>
            {error && <ErrorBox message={error} />}
            <button type="submit" disabled={fetching} className="btn-primary w-full py-3 flex justify-center items-center gap-2">
              {fetching ? <><Loader2 className="w-5 h-5 animate-spin" />Fetching post...</> : <><Globe className="w-5 h-5" />Fetch Post</>}
            </button>
            <p className="text-xs text-gray-500 text-center">
              Extraction is limited to content that the configured backend extractor can access. It does not bypass private accounts, login walls or platform restrictions.
            </p>
          </form>
        ) : (
          <div className="space-y-6">
            <div className="card">
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="px-3 py-1 rounded-full bg-primary-50 text-primary-700 text-sm font-medium">{post.platform}</span>
                {post.extraction_status && <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-700 text-sm">{post.extraction_status}</span>}
                {post.extraction_method && <span className="text-xs text-gray-500">via {post.extraction_method}</span>}
              </div>

              {post.image_url && (
                <img src={post.image_url} alt="Extracted social-media content" className="max-h-96 max-w-full mx-auto rounded-lg object-contain mb-5" />
              )}

              <div className="space-y-2 text-sm text-gray-600">
                {post.author && <p><strong>Author:</strong> {post.author}</p>}
                {post.published_at && <p><strong>Published:</strong> {post.published_at}</p>}
                {post.location && <p><strong>Location:</strong> {post.location}</p>}
              </div>

              <a href={post.canonical_url || post.post_url} target="_blank" rel="noopener noreferrer" className="mt-4 inline-flex items-center gap-2 text-primary-600 hover:underline text-sm">
                <ExternalLink className="w-4 h-4" /> View original post
              </a>
            </div>

            <div className="card">
              <h2 className="text-lg font-semibold mb-2">Claim to Verify</h2>
              <textarea
                value={claim}
                onChange={(event: { target: { value: string } }) => setClaim(event.target.value)}
                rows={5}
                disabled={verifying}
                className="input-field"
                placeholder="Enter or edit the factual claim extracted from the post..."
              />
              <p className="text-xs text-gray-500 mt-2">
                The original post URL and extraction metadata are preserved as case provenance.
              </p>
            </div>

            {error && <ErrorBox message={error} />}

            <div className="flex flex-col sm:flex-row gap-3">
              <button type="button" className="btn-secondary" onClick={() => { setPost(null); setClaim(''); setStatus(''); setError(null) }} disabled={verifying}>
                Fetch another post
              </button>
              <button type="button" className="btn-primary flex-1 py-3 flex justify-center items-center gap-2" onClick={() => void verify()} disabled={verifying || !claim.trim()}>
                {verifying ? <><Loader2 className="w-5 h-5 animate-spin" />{status || 'Verifying...'}</> : <><CheckCircle className="w-5 h-5" />Verify This Post</>}
              </button>
            </div>

            {status === 'Verification complete' && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-green-700 flex items-center gap-2">
                <CheckCircle className="w-5 h-5" /> Verification complete. Opening the case...
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex gap-3">
      <ShieldAlert className="w-5 h-5 flex-shrink-0" />
      <span>{message}</span>
    </div>
  )
}
