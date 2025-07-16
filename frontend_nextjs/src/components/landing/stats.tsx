const stats = [
  { id: 1, name: "Hours saved per week", value: "20+" },
  { id: 2, name: "Reduction in ad-hoc requests", value: "90%" },
  { id: 3, name: "Faster stakeholder insights", value: "10x" },
  // { id: 4, name: "Paid out to creators", value: "$70M" },
];

export default function Example() {
  return (
    <div className="bg-gray-900 py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl lg:max-w-none">
          <div className="text-center">
            <h2 className="text-4xl font-semibold tracking-tight text-balance text-gray-50 sm:text-5xl">
              The result?
            </h2>
            <p className="mt-4 text-lg/8 text-gray-300">
              Fewer bottlenecks, faster insights, more impact.
            </p>
          </div>
          <dl className="mt-16 grid grid-cols-1 gap-0.5 overflow-hidden rounded-2xl text-center sm:grid-cols-3">
            {stats.map((stat) => (
              <div key={stat.id} className="flex flex-col bg-gray-800 p-8">
                <dt className="text-sm/6 font-semibold text-gray-300">
                  {stat.name}
                </dt>
                <dd className="order-first text-3xl font-semibold tracking-tight text-gray-50">
                  {stat.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}
