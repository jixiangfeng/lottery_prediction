<template>
  <span class="ball" :class="numberZone(value)">{{ String(value).padStart(2, '0') }}</span>
</template>

<script setup lang="ts">
import { numberZone } from '../domain/report'

defineProps<{ value: number }>()
</script>
