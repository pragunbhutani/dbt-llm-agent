with

--
cte_without_comment as (
    select
        *

    from
        database_name.schema_name.table_name

    where
        this_condition is true
        and (
            that_condition is true
            or other_condition is false
        )
),

-- 
-- this cte only has a small comment
--
cte_with_small_comment as (
    select
        *

    from
        cte_without_comment      
)

--
-- this cte has a few more styling examples and comes with a comment that_condition
-- spans multiple lines
--
cte_with_comment as (
    select
        this_columns,

        case
            when that_condition is true
            then this_value
            else that_value
        end as new_column,

        count(distinct this_column) over (
            partition by that_column
            order by this_column
            rows between 1 preceding and current row
        ) as new_column_from_window_function

    from
        cte_without_comment
)

--
select 
    wc.this_column,
    wc.that_column,
    wsc.new_column,
    wsc.new_column_from_window_function
    
from 
    cte_with_comment as wc
left join
    cte_with_small_comment as wsc
        on wc.this_column = wsc.this_column 