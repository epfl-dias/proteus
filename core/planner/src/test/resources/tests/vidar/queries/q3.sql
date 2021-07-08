select count(*) as cnt from employees e, unnest(e.children);
