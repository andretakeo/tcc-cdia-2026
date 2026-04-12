"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Bar,
  ComposedChart,
} from "recharts";

interface ChartData {
  date: string;
  price: number;
  sentiment: number;
}

export function SentimentChart({ data }: { data: ChartData[] }) {
  return (
    <div className="h-[400px] w-full bg-white p-4 rounded-xl border shadow-sm">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }} 
            minTickGap={30}
          />
          <YAxis 
            yAxisId="left" 
            orientation="left" 
            stroke="#8884d8" 
            domain={['auto', 'auto']}
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            yAxisId="right" 
            orientation="right" 
            stroke="#82ca9d" 
            domain={[-1, 1]}
            tick={{ fontSize: 12 }}
          />
          <Tooltip 
            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
          />
          <Legend verticalAlign="top" align="right" height={36}/>
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="price"
            stroke="#8884d8"
            name="Preço (R$)"
            dot={false}
            strokeWidth={2}
          />
          <Bar
            yAxisId="right"
            dataKey="sentiment"
            fill="#82ca9d"
            name="Sentimento AI"
            opacity={0.6}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
