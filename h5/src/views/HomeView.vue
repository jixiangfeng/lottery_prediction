<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import NumberBall from '../components/NumberBall.vue'
import { fetchReportCatalog } from '../api/catalog'
import { useReportStore } from '../stores/reportStore'
import { selectReportIssue } from '../domain/catalog'

const route = useRoute()
const router = useRouter()
const { state, loadReport } = useReportStore()
const catalog = ref<{ latestIssue: string; reports: Array<{ issue: string; label: string; path: string }> }>({ latestIssue: '', reports: [] })

const selectedIssue = computed(() => selectReportIssue(catalog.value, route.query.issue as string | undefined))

async function syncSelection(issue?: string) {
  const resolved = issue ?? selectedIssue.value
  await loadReport(resolved || undefined, true)
}

function chooseIssue(issue: string) {
  void router.push({ name: 'report-detail', params: { issue } })
}

onMounted(async () => {
  catalog.value = await fetchReportCatalog()
  await syncSelection(route.query.issue as string | undefined)
})

watch(
  () => route.query.issue,
  (issue) => {
    void syncSelection(issue as string | undefined)
  },
)
</script>

<template>
  <section v-if="state.loading" class="state-card">正在加载今日报告...</section>
  <section v-else-if="state.error" class="state-card error">{{ state.error }}</section>
  <template v-else-if="state.report">
    <section class="hero-card blue-hero">
      <p class="eyebrow">快乐8 H5 用户端</p>
      <h1>历史日报</h1>
      <p class="subline">蓝白风格 · 可切换历史期号查看分析结果</p>
    </section>

    <section class="panel">
      <header>
        <div>
          <p class="eyebrow">Reports</p>
          <h2>历史期号</h2>
        </div>
        <span class="pill">共 {{ catalog.reports.length }} 期</span>
      </header>
      <div class="history-list">
        <button
          v-for="item in catalog.reports"
          :key="item.issue"
          class="history-item"
          :class="{ active: item.issue === state.report.issue }"
          @click="chooseIssue(item.issue)"
        >
          <strong>{{ item.issue }}</strong>
          <span>{{ item.label }}</span>
        </button>
      </div>
    </section>

    <section class="summary-grid">
      <article class="metric-card">
        <span>最新期号</span>
        <strong>{{ state.report.issue }}</strong>
      </article>
      <article class="metric-card">
        <span>主推荐参数</span>
        <strong>{{ state.report.parameterName }}</strong>
      </article>
      <article class="metric-card">
        <span>策略模式</span>
        <strong>{{ state.report.strategyMode ?? 'auto' }}</strong>
      </article>
      <article class="metric-card">
        <span>最近回测 ROI</span>
        <strong :class="{ loss: (state.report.backtest?.roi ?? 0) < 0 }">{{ state.report.backtest?.roiPercent }}</strong>
      </article>
      <article class="metric-card">
        <span>数据质量</span>
        <strong :class="{ loss: state.report.dataQuality?.ok === false }">
          {{ state.report.dataQuality?.ok === false ? '异常' : '正常' }}
        </strong>
      </article>
      <article class="metric-card">
        <span>数据源</span>
        <strong :class="{ loss: state.report.dataSource?.usedCache }">
          {{ state.report.dataSource?.usedCache ? '本地缓存' : '官方同步' }}
        </strong>
      </article>
      <article class="metric-card">
        <span>实盘加权</span>
        <strong>{{ state.report.liveParameterWeights?.enabled ? '启用' : '未启用' }}</strong>
      </article>
      <article class="metric-card">
        <span>前推加权</span>
        <strong>{{ state.report.walkForwardParameterWeights?.enabled ? state.report.walkForwardParameterWeights.bestStrategy ?? '启用' : '未启用' }}</strong>
      </article>
    </section>

    <section class="panel">
      <header>
        <div>
          <p class="eyebrow">Latest Draw</p>
          <h2>最新开奖号码</h2>
        </div>
      </header>
      <div class="balls">
        <NumberBall v-for="num in state.report.latestNumbers" :key="num" :value="num" />
      </div>
    </section>
  </template>
</template>
