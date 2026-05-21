export default function MessageSkeleton() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-4 animate-pulse">
      {/* User message */}
      <div className="flex justify-end">
        <div className="w-2/3 h-10 bg-slate-200 rounded-2xl rounded-br-md" />
      </div>
      {/* Assistant message */}
      <div className="flex justify-start gap-3">
        <div className="w-8 h-8 rounded-full bg-slate-200 shrink-0" />
        <div className="space-y-2 flex-1">
          <div className="h-4 bg-slate-200 rounded w-3/4" />
          <div className="h-4 bg-slate-200 rounded w-1/2" />
          <div className="h-4 bg-slate-200 rounded w-5/6" />
        </div>
      </div>
      {/* Another assistant line */}
      <div className="flex justify-start gap-3">
        <div className="w-8 h-8 shrink-0" />
        <div className="space-y-2 flex-1">
          <div className="h-4 bg-slate-200 rounded w-2/3" />
          <div className="h-4 bg-slate-200 rounded w-1/3" />
        </div>
      </div>
    </div>
  );
}
