import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CheckCircle, ImagePlus, Loader2, ShieldAlert, X } from 'lucide-react'
import { getApiError, verificationApi } from '../services/api'

export function VerifyLocalPage() {
  const navigate = useNavigate()
  const [claim, setClaim] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [promoteKnowledge, setPromoteKnowledge] = useState(false)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('Ready')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  const chooseImage = (file: File | undefined) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Image must be smaller than 10 MB.')
      return
    }
    setImageFile(file)
    setPreview(URL.createObjectURL(file))
    setError(null)
  }

  const submit = async (event: any) => {
    event.preventDefault()
    if (!claim.trim()) return setError('Please enter a claim.')
    if (!imageFile) return setError('Please upload an image.')

    setLoading(true)
    setError(null)
    setStatus('Preparing image...')

    try {
      const imageData = await fileToBase64(imageFile)
      setStatus('Running verification pipeline...')
      const jobResult = await verificationApi.start({
        claim: claim.trim(),
        image_path: imageData,
        promote_knowledge: promoteKnowledge,
        metadata: {
          source_platform: 'local',
          original_filename: imageFile.name,
          mime_type: imageFile.type,
          file_size: imageFile.size,
        },
      })

      setStatus('Verification in progress. Polling for result...')
      // Poll for completion instead of waiting synchronously
      const final = await verificationApi.poll(jobResult.job_id, (s) =>
        setStatus(`Running: ${s.current_stage || 'queued'}${s.stage_status ? ` (${s.stage_status})` : ''} - this can take 15-20 minutes on a CPU`))
      
      if (final.status === 'failed') {
        throw new Error(final.error || 'Verification job failed.')
      }
      
      if (!final.case_id) throw new Error('The backend completed without returning a case ID.')
      
      setStatus('Verification complete. Opening case...')
      window.setTimeout(() => navigate(`/case/${final.case_id}`), 500)
    } catch (err) {
      setError(getApiError(err, 'Verification failed.'))
      setStatus('Verification failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <ShieldAlert className="w-12 h-12 text-primary-600 mx-auto mb-3" />
          <h1 className="text-3xl font-bold">Verify Local Content</h1>
          <p className="text-gray-600 mt-2">Submit an image and a claim to run the complete verification workflow.</p>
        </div>

        <form onSubmit={submit} className="card space-y-6">
          <div>
            <label htmlFor="local-image" className="block text-sm font-medium mb-2">Image <span className="text-red-500">*</span></label>
            <input
              id="local-image"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              disabled={loading}
              className="hidden"
              onChange={(event: { target: { files?: FileList | null } }) => chooseImage(event.target.files?.[0])}
            />
            {preview ? (
              <div className="relative border rounded-xl p-3 bg-gray-50">
                <img src={preview} alt="Selected verification input" className="max-h-80 max-w-full mx-auto rounded-lg object-contain" />
                <button
                  type="button"
                  onClick={() => { setImageFile(null); setPreview(null) }}
                  className="absolute top-3 right-3 p-2 bg-red-600 text-white rounded-full"
                  aria-label="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <label htmlFor="local-image" className="border-2 border-dashed border-gray-300 hover:border-primary-400 rounded-xl p-10 flex flex-col items-center cursor-pointer">
                <ImagePlus className="w-10 h-10 text-gray-400 mb-3" />
                <span className="font-medium text-gray-700">Choose an image</span>
                <span className="text-sm text-gray-500 mt-1">PNG, JPG or WebP · maximum 10 MB</span>
              </label>
            )}
          </div>

          <div>
            <label htmlFor="claim" className="block text-sm font-medium mb-2">Claim to Verify <span className="text-red-500">*</span></label>
            <textarea
              id="claim"
              value={claim}
              maxLength={2000}
              rows={5}
              disabled={loading}
              onChange={(event: { target: { value: string } }) => setClaim(event.target.value)}
              className="input-field"
              placeholder="Enter the specific factual claim associated with the image..."
            />
            <div className="text-xs text-gray-500 text-right mt-1">{claim.length}/2000</div>
          </div>

          <label className="flex gap-3 items-start text-sm text-gray-700">
            <input
              type="checkbox"
              checked={promoteKnowledge}
              onChange={(event: { target: { checked: boolean } }) => setPromoteKnowledge(event.target.checked)}
              disabled={loading}
              className="mt-1"
            />
            <span>
              <strong>Promote validated KG candidates</strong>
              <span className="block text-gray-500 mt-0.5">Only use this when you want validated candidates from this case considered for permanent knowledge-graph promotion.</span>
            </span>
          </label>

          {error && (
            <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 flex gap-3">
              <ShieldAlert className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button type="submit" disabled={loading || !claim.trim() || !imageFile} className="btn-primary w-full py-3 flex items-center justify-center gap-2">
            {loading ? <><Loader2 className="w-5 h-5 animate-spin" />{status}</> : <><CheckCircle className="w-5 h-5" />Start Verification</>}
          </button>

          {status === 'Verification complete. Opening case...' && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg text-green-700 flex items-center gap-2">
              <CheckCircle className="w-5 h-5" /> Verification completed successfully.
            </div>
          )}
        </form>

        <div className="card mt-6">
          <h2 className="text-lg font-semibold mb-3">Pipeline</h2>
          <p className="text-sm text-gray-600">
            The backend runs preprocessing, multimodal retrieval, Knowledge Graph retrieval, live evidence retrieval,
            narrative fusion, specialist verification, uncertainty analysis, adaptive debate, judging, explainability and KG expansion.
          </p>
        </div>
      </div>
    </div>
  )
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('Unable to read the selected image.'))
    reader.readAsDataURL(file)
  })
}
