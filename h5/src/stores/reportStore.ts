import { reactive } from 'vue'
import { fetchLatestReport } from '../api/report'
import type { ReportData } from '../domain/report'

interface ReportState {
  report: ReportData | null
  loading: boolean
  error: string
  loadedIssue: string | null
}

const state = reactive<ReportState>({
  report: null,
  loading: false,
  error: '',
  loadedIssue: null,
})

export async function loadReport(issue?: string, force = false): Promise<ReportData | null> {
  if (!force && state.loadedIssue === (issue ?? null) && state.report) {
    return state.report
  }
  state.loading = true
  state.error = ''
  try {
    state.report = await fetchLatestReport(issue)
    state.loadedIssue = issue ?? null
    return state.report
  } catch (err) {
    state.error = err instanceof Error ? err.message : '读取数据失败'
    return null
  } finally {
    state.loading = false
  }
}

export function useReportStore() {
  return {
    state,
    loadReport,
  }
}
