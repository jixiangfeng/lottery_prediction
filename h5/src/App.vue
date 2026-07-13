<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import { navItems } from './router'
</script>

<template>
  <main class="app-shell">
    <RouterView />
  </main>

  <nav class="bottom-tabs" aria-label="用户端底部导航">
    <RouterLink v-for="item in navItems" :key="item.to" :to="item.to" class="tab-item">
      <span class="tab-icon">{{ item.icon }}</span>
      <span>{{ item.label }}</span>
    </RouterLink>
  </nav>
</template>
