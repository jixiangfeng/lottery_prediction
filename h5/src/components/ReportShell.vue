<script setup lang="ts">
import { onMounted } from 'vue'
import { useReportStore } from '../stores/reportStore'

const { state, loadReport } = useReportStore()
onMounted(() => void loadReport())
</script>

<template>
  <section v-if="state.loading" class="state-card">正在加载今日报告...</section>
  <section v-else-if="state.error" class="state-card error">{{ state.error }}</section>
  <slot v-else-if="state.report" :report="state.report" />
</template>
